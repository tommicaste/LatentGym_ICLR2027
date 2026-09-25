"""
BenchmarkOrchestrator — Flat orchestration over (model, env_config) pairs.

Single-agent eval:
    - API models (OpenAI, Anthropic, etc.) → APIRunner
    - Local models → use SkyRL pipeline directly (see LocalRunner for setup)

Double-agent eval:
    - Both API and local models → APIRunner (SkyRL doesn't support agent switching)

W&B tracking is optional — pass wandb_config to enable.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from latentgym.core.env_config import FullyDefinedEnv
from latentgym.core.registry import make_env
from .model_interface import ModelInterface
from .results import BenchmarkResults
from .single_agent.api_runner import APIRunner
from .double_agent.runner import DoubleAgentRunner
from .wandb_tracker import WandbConfig, WandbTracker, DummyTracker, create_tracker

logger = logging.getLogger(__name__)


class BenchmarkOrchestrator:
    """Flat orchestration: iterates over (model, env_config) pairs.

    Handles progress tracking, checkpointing, and optional W&B logging.

    For single-agent eval with API models, uses APIRunner directly.
    For local models, use LocalRunner to build parquet + run via SkyRL pipeline,
    or use APIRunner with VLLMModel for simpler (non-batched) evaluation.

    For double-agent eval, uses APIRunner for both model types
    (SkyRL's generator doesn't support mid-trajectory agent switching).
    """

    def __init__(
        self,
        models: List[ModelInterface],
        env_configs: List[FullyDefinedEnv],
        trajectory_dir: str,
        n_trajectories_per_config: int = 50,
        seed: int = 42,
        checkpoint_dir: Optional[str] = None,
        max_total_turns: int = 500,
        wandb_config: Optional[WandbConfig] = None,
        output_dir: Optional[str] = None,
        start_trajectory: int = 0,
    ):
        """
        Args:
            models: List of models to evaluate
            env_configs: List of fully defined environments
            trajectory_dir: Directory containing trajectory files (manifest.json + traj_*.json).
                Each env_config should have a corresponding subdirectory.
            n_trajectories_per_config: Trajectories per (model, env) pair
            seed: Base random seed
            checkpoint_dir: Directory for checkpointing (resume support)
            max_total_turns: Safety limit per trajectory
            wandb_config: W&B config to enable tracking, None to disable
            output_dir: Directory to incrementally save trajectories as they complete.
                If None, trajectories are only saved at the end via report.save().
        """
        self.models = models
        self.env_configs = env_configs
        self.trajectory_dir = trajectory_dir
        self.n_trajectories = n_trajectories_per_config
        self.seed = seed
        self.checkpoint_dir = checkpoint_dir
        self.max_total_turns = max_total_turns
        self.output_dir = output_dir
        self.start_trajectory = start_trajectory
        self.results = BenchmarkResults()
        self.tracker = create_tracker(wandb_config)

        if checkpoint_dir:
            self._load_checkpoint()

    async def run(self) -> BenchmarkResults:
        """Run all (model × env_config) combinations using APIRunner.

        Loads trajectory files from trajectory_dir and runs each model
        against each env configuration. Optionally logs to W&B.
        """
        # Init tracker with run config
        self.tracker.init(run_config={
            "models": [m.name for m in self.models],
            "env_configs": [c.benchmark_id for c in self.env_configs],
            "n_trajectories": self.n_trajectories,
            "seed": self.seed,
            "mode": "single_agent",
        })

        total = len(self.models) * len(self.env_configs)
        completed = 0

        for model in self.models:
            runner = APIRunner(model)

            for config in self.env_configs:
                bid = config.benchmark_id

                if self._already_completed(model.name, bid):
                    completed += 1
                    logger.info(f"[{completed}/{total}] Skipping {model.name} × {bid} (cached)")
                    continue

                completed += 1
                logger.info(f"[{completed}/{total}] Running {model.name} × {bid} "
                           f"({self.n_trajectories} trajectories)")

                # Find trajectory files for this config
                try:
                    traj_files = self._get_trajectory_files(config)
                except FileNotFoundError as e:
                    logger.warning(f"  Skipping — data not found: {e}")
                    continue

                trajectories = []
                for i, traj_path in enumerate(traj_files[:self.n_trajectories]):
                    if i < self.start_trajectory:
                        logger.info(f"  Trajectory {i+1}/{self.n_trajectories} skipped (--start-trajectory {self.start_trajectory})")
                        continue
                    env = make_env(config, trajectory_path=traj_path)
                    result = await runner.run_trajectory(
                        env, seed=self.seed + i,
                        max_total_turns=self.max_total_turns,
                    )
                    result.benchmark_id = bid
                    result.prompt_id = config.prompt_id
                    result.feedback_id = config.feedback_id

                    # Save trajectory to disk immediately
                    self._save_trajectory(model.name, bid, i, result)
                    logger.info(f"  Trajectory {i+1}/{self.n_trajectories} done "
                               f"(reward={result.cumulative_reward:.3f}, "
                               f"episodes={result.num_episodes})")

                    # Log per-episode and per-trajectory to wandb
                    for outcome in result.episode_outcomes:
                        self.tracker.log_episode(model.name, bid, i, outcome)
                    self.tracker.log_trajectory(model.name, bid, i, result)

                    # Free the heavy data after saving to disk — keep only
                    # episode_outcomes for in-memory metrics, not full conversation
                    result.conversation = []
                    result.reasoning_trace = []
                    trajectories.append(result)

                self.results.add(model.name, bid, trajectories)
                self._save_checkpoint()

        self.tracker.finish()
        return self.results

    async def run_double_agent(
        self,
        model_a: ModelInterface,
        model_b: ModelInterface,
        switch_episode: int,
    ) -> BenchmarkResults:
        """Run double-agent evaluation.

        Uses APIRunner for both model types since SkyRL's generator
        doesn't support mid-trajectory agent switching.
        """
        schedule_name = f"{model_a.name}→{model_b.name}@ep{switch_episode}"

        self.tracker.init(run_config={
            "model_a": model_a.name,
            "model_b": model_b.name,
            "switch_episode": switch_episode,
            "env_configs": [c.benchmark_id for c in self.env_configs],
            "n_trajectories": self.n_trajectories,
            "seed": self.seed,
            "mode": "double_agent",
        })

        runner = DoubleAgentRunner(model_a, model_b)
        double_results = BenchmarkResults()

        for config in self.env_configs:
            bid = config.benchmark_id
            logger.info(f"Double-agent: {schedule_name} × {bid}")

            try:
                traj_files = self._get_trajectory_files(config)
            except FileNotFoundError as e:
                logger.warning(f"  Skipping — data not found: {e}")
                continue

            trajectories = []
            for i, traj_path in enumerate(traj_files[:self.n_trajectories]):
                env = make_env(config, trajectory_path=traj_path)
                result = await runner.run_trajectory(
                    env, switch_episode=switch_episode,
                    seed=self.seed + i,
                    max_total_turns=self.max_total_turns,
                )
                result.benchmark_id = bid
                result.prompt_id = config.prompt_id
                result.feedback_id = config.feedback_id

                # Save trajectory to disk immediately
                self._save_trajectory(result.model_name, bid, i, result)
                logger.info(f"  Trajectory {i+1}/{self.n_trajectories} done "
                           f"(reward={result.cumulative_reward:.3f})")

                # Log per-episode and per-trajectory to wandb
                for outcome in result.episode_outcomes:
                    self.tracker.log_episode(result.model_name, bid, i, outcome)
                self.tracker.log_trajectory(result.model_name, bid, i, result)

                # Free heavy data after saving to disk
                result.conversation = []
                result.reasoning_trace = []
                trajectories.append(result)

            double_results.add(result.model_name, bid, trajectories)

        self.tracker.finish()
        return double_results

    def _get_trajectory_files(self, config: FullyDefinedEnv) -> List[str]:
        """Find trajectory files for a config in trajectory_dir."""
        from latentgym.core.trajectory_utils import load_manifest

        base = Path(self.trajectory_dir)

        # Try nested directory: data/eval/{env_name}/{latent_id}/
        subdir = base / config.env_name / config.latent_id
        if not subdir.exists():
            # Try flat directory: data/eval/{env_name}_{latent_id}/
            subdir = base / f"{config.env_name}_{config.latent_id}"
        if not subdir.exists():
            # Try base directory directly
            subdir = base

        manifest = load_manifest(str(subdir))
        return [str(subdir / f) for f in manifest.trajectory_files]

    def _already_completed(self, model_name: str, benchmark_id: str) -> bool:
        existing = self.results.get(model_name, benchmark_id)
        return len(existing) >= self.n_trajectories

    def _save_trajectory(self, model_name: str, benchmark_id: str, idx: int, result):
        """Save a single trajectory to disk immediately after completion."""
        if not self.output_dir:
            return
        import json
        safe_model = model_name.replace("/", "__").replace(":", "_")
        safe_bid = benchmark_id.replace("/", "__")
        dest = Path(self.output_dir) / "trajectories" / safe_model / safe_bid
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / f"traj_{idx:04d}.json"
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

    def _save_checkpoint(self):
        if not self.checkpoint_dir:
            return
        path = Path(self.checkpoint_dir) / "checkpoint.json"
        self.results.save(str(path))
        logger.debug(f"Checkpoint saved to {path}")

    def _load_checkpoint(self):
        if not self.checkpoint_dir:
            return
        path = Path(self.checkpoint_dir) / "checkpoint.json"
        if path.exists():
            self.results = BenchmarkResults.load(str(path))
            logger.info(f"Resumed from checkpoint: {path}")
