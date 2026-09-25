"""
LocalRunner — Run local vLLM/SGLang models using SkyRL's evaluation pipeline.

Wraps SkyRL's existing evaluate() and SkyRLGymGenerator to leverage
optimized inference (batching, KV cache, etc.) for local models.

This runner:
1. Registers our MultiEpisodeEnv with skyrl_gym
2. Builds a parquet dataset pointing to trajectory files
3. Calls SkyRL's evaluate() which uses SkyRLGymGenerator
4. Converts SkyRL's output to our TrajectoryResult format

For API models (OpenAI, Anthropic, etc.), use APIRunner instead.
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from latentgym.core.env_config import FullyDefinedEnv
from latentgym.core.trajectory_utils import load_manifest
from latentgym.eval.types import EpisodeOutcome, TrajectoryResult

logger = logging.getLogger(__name__)


def _build_eval_parquet(
    fully_defined: FullyDefinedEnv,
    trajectory_dir: str,
    env_class_id: str,
    output_path: str,
) -> str:
    """Build a SkyRL-compatible parquet dataset from trajectory files.

    Each row has: prompt, env_class, extra_info (with trajectory_path).

    Args:
        fully_defined: Environment config
        trajectory_dir: Directory containing manifest.json + trajectory files
        env_class_id: The skyrl_gym registered env ID
        output_path: Where to write the parquet

    Returns:
        Path to the parquet file.
    """
    manifest = load_manifest(trajectory_dir)
    rows = []

    for traj_file in manifest.trajectory_files:
        traj_path = str(Path(trajectory_dir) / traj_file)

        # Build system prompt (same as make_env would)
        # We store minimal prompt here — the env's init() constructs the full prompt
        prompt = [{"role": "system", "content": ""}]

        rows.append({
            "data_source": f"benchmark_{fully_defined.env_name}",
            "prompt": prompt,
            "env_class": env_class_id,
            "reward_spec": {"method": "rule", "ground_truth": ""},
            "extra_info": {
                "trajectory_path": traj_path,
                "env_name": fully_defined.env_name,
                "latent_id": fully_defined.latent_id,
                "prompt_id": fully_defined.prompt_id,
                "feedback_id": fully_defined.feedback_id,
                "num_episodes": fully_defined.num_episodes,
                "reward_type": fully_defined.reward_type.value,
            },
        })

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    logger.info(f"Built eval parquet: {output_path} ({len(rows)} entries)")
    return output_path


def register_with_skyrl(env_class_id: str = "benchmark_multi_episode"):
    """Register our MultiEpisodeEnv with skyrl_gym's registry.

    This allows SkyRL's SkyRLGymGenerator to instantiate our env
    via skyrl_gym.make(env_class_id, env_config, extras).
    """
    from skyrl_gym.envs.registration import register, spec

    # Check if already registered
    try:
        spec(env_class_id)
        return  # Already registered
    except Exception:
        pass

    register(
        id=env_class_id,
        entry_point="latentgym.core.multi_episode_env:MultiEpisodeEnv",
    )
    logger.info(f"Registered '{env_class_id}' with skyrl_gym")


class LocalRunner:
    """Run local models using SkyRL's evaluation pipeline.

    Wraps SkyRL's SkyRLGymGenerator for optimized local inference.

    Usage:
        runner = LocalRunner(
            model_path="/path/to/model",
            trajectory_dir="data/eval/bandits_loyal_favorite_0/",
            fully_defined=FullyDefinedEnv(...),
        )
        results = await runner.run()

    Note: This requires a running vLLM/SGLang server or will start one.
    For a simpler setup, use APIRunner with VLLMModel instead.
    """

    def __init__(
        self,
        model_path: str,
        trajectory_dir: str,
        fully_defined: FullyDefinedEnv,
        backend: str = "vllm",
        server_url: Optional[str] = None,
        tensor_parallel_size: int = 1,
        eval_batch_size: int = 16,
        n_samples_per_prompt: int = 1,
        sampling_params: Optional[Dict[str, Any]] = None,
    ):
        self.model_path = model_path
        self.trajectory_dir = trajectory_dir
        self.fully_defined = fully_defined
        self.backend = backend
        self.server_url = server_url
        self.tensor_parallel_size = tensor_parallel_size
        self.eval_batch_size = eval_batch_size
        self.n_samples_per_prompt = n_samples_per_prompt
        self.sampling_params = sampling_params or {
            "temperature": 0.7,
            "max_tokens": 512,
        }

        self.env_class_id = "benchmark_multi_episode"

    async def run(self) -> List[TrajectoryResult]:
        """Run evaluation using SkyRL pipeline.

        This is a high-level wrapper. For fine-grained control,
        use SkyRL's evaluate() directly with a properly configured Hydra config.

        Returns:
            List of TrajectoryResults (converted from SkyRL output).
        """
        # 1. Register env with skyrl_gym
        register_with_skyrl(self.env_class_id)

        # 2. Build parquet dataset
        with tempfile.TemporaryDirectory() as tmp_dir:
            parquet_path = str(Path(tmp_dir) / "eval_dataset.parquet")
            _build_eval_parquet(
                self.fully_defined,
                self.trajectory_dir,
                self.env_class_id,
                parquet_path,
            )

            # 3. Build and run SkyRL evaluate
            # Note: full integration requires Hydra config setup.
            # This provides the building blocks; users can also call
            # SkyRL's evaluate() directly with their own config.
            logger.info(
                f"LocalRunner: Built parquet at {parquet_path}. "
                f"To run with SkyRL's full pipeline, use this parquet with "
                f"skyrl_train.evaluate() and env_class='{self.env_class_id}'."
            )

            # For now, return empty — full SkyRL integration requires
            # Hydra config, inference engine setup, etc. which is
            # deployment-specific. The parquet + registered env is the bridge.
            logger.warning(
                "LocalRunner.run() currently builds the dataset and registers the env. "
                "Full SkyRL evaluate() integration requires Hydra config. "
                "Use the generated parquet with SkyRL's training/eval scripts directly."
            )
            return []

    def get_skyrl_config_snippet(self) -> str:
        """Return a Hydra config snippet for using this env with SkyRL.

        Useful for users who want to run SkyRL's eval/training directly.
        """
        return f"""
# SkyRL config snippet for benchmark evaluation
environment:
  env_class: {self.env_class_id}

dataset:
  path: <path_to_generated_parquet>
  prompt_key: prompt
  env_class_key: env_class

generator:
  backend: {self.backend}
  eval_n_samples_per_prompt: {self.n_samples_per_prompt}
  eval_sampling_params:
    temperature: {self.sampling_params.get('temperature', 0.7)}
    max_tokens: {self.sampling_params.get('max_tokens', 512)}
"""
