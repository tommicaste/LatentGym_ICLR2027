"""
Core data structures for evaluation results.

Captures all data generated during trajectory runs:
- Per-episode: reward, turns, latent_id, agent, outcome type
- Per-trajectory: full conversation, env config, all episode outcomes
- Full conversation stored once (not duplicated per-step)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class OutcomeType(Enum):
    """How an episode ended."""
    WIN = "win"
    LOSS = "loss"
    TIMEOUT = "timeout"
    PARTIAL = "partial"


@dataclass
class EpisodeOutcome:
    """Result of a single episode within a trajectory."""
    episode_idx: int
    reward: float
    turns: int
    success: bool
    agent_name: str

    # Episode context
    latent_id: str = ""
    max_turns: int = 0
    outcome_type: OutcomeType = OutcomeType.PARTIAL

    # Ground truth for this episode (e.g., target_word, ground_truth probs, draws)
    # Populated from the episode_config that was used to reset the env
    ground_truth: Dict[str, Any] = field(default_factory=dict)

    # Generic metadata (env-specific extras)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def turn_efficiency(self) -> float:
        """1.0 = used minimum turns, 0.0 = used all turns."""
        if self.max_turns <= 0:
            return 0.0
        return 1.0 - (self.turns / self.max_turns)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_idx": self.episode_idx,
            "reward": self.reward,
            "turns": self.turns,
            "success": self.success,
            "agent_name": self.agent_name,
            "latent_id": self.latent_id,
            "max_turns": self.max_turns,
            "outcome_type": self.outcome_type.value,
            "turn_efficiency": self.turn_efficiency,
            "ground_truth": self.ground_truth,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> EpisodeOutcome:
        return cls(
            episode_idx=d["episode_idx"],
            reward=d["reward"],
            turns=d["turns"],
            success=d["success"],
            agent_name=d["agent_name"],
            latent_id=d.get("latent_id", ""),
            max_turns=d.get("max_turns", 0),
            outcome_type=OutcomeType(d.get("outcome_type", "partial")),
            ground_truth=d.get("ground_truth", {}),
            metadata=d.get("metadata", {}),
        )


@dataclass
class TrajectoryResult:
    """Result of running a complete multi-episode trajectory.

    Captures everything: env config, all episode outcomes, full conversation,
    and all metadata from init and final step.

    The full conversation is stored once as a flat list of messages.
    Per-episode actions/feedback can be reconstructed from conversation +
    episode_outcomes (which records turn counts per episode).

    Reasoning/thinking is stored separately from the conversation. Each entry
    in `reasoning_trace` corresponds to an assistant turn (in order). Only
    turns where the model produced reasoning have non-None entries.
    Reasoning is NOT included in conversation — only the action text is.
    """
    # Core data
    episode_outcomes: List[EpisodeOutcome]
    conversation: List[Dict[str, str]]

    # Identity
    model_name: str
    benchmark_id: str = ""
    seed: int = 0

    # Environment config (from FullyDefinedEnv)
    env_name: str = ""
    latent_id: str = ""
    prompt_id: str = ""
    feedback_id: str = ""
    reward_type: str = ""

    # Environment params
    max_turns_per_episode: int = 0
    env_params: Dict[str, Any] = field(default_factory=dict)

    # Agent assignments (who played each episode)
    agent_assignments: List[str] = field(default_factory=list)

    # Per-episode ground truth configs (target_word, ground_truth probs, draws, etc.)
    episode_configs: List[Dict[str, Any]] = field(default_factory=list)

    # Per-turn reasoning/thinking from models that support it (OpenAI o-series via
    # Responses API, Claude extended thinking, Gemini thinking, vLLM reasoning models).
    # One entry per assistant turn, in order. None for turns without reasoning.
    # Reasoning is recorded here but NOT included in conversation context.
    reasoning_trace: List[Optional[str]] = field(default_factory=list)

    # Init metadata (everything from env.init())
    init_metadata: Dict[str, Any] = field(default_factory=dict)

    # Final step metadata (everything from last env.step())
    final_metadata: Dict[str, Any] = field(default_factory=dict)

    # Generic extra metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Derived properties ──

    @property
    def episode_rewards(self) -> List[float]:
        return [o.reward for o in self.episode_outcomes]

    @property
    def episode_turns(self) -> List[int]:
        return [o.turns for o in self.episode_outcomes]

    @property
    def num_episodes(self) -> int:
        return len(self.episode_outcomes)

    @property
    def cumulative_reward(self) -> float:
        return sum(self.episode_rewards)

    @property
    def terminal_reward(self) -> float:
        return self.episode_rewards[-1] if self.episode_rewards else 0.0

    @property
    def initial_reward(self) -> float:
        return self.episode_rewards[0] if self.episode_rewards else 0.0

    @property
    def improvement(self) -> float:
        if len(self.episode_rewards) < 2:
            return 0.0
        return self.episode_rewards[-1] - self.episode_rewards[0]

    @property
    def mean_reward(self) -> float:
        if not self.episode_rewards:
            return 0.0
        return sum(self.episode_rewards) / len(self.episode_rewards)

    @property
    def success_rate(self) -> float:
        if not self.episode_outcomes:
            return 0.0
        return sum(1 for o in self.episode_outcomes if o.success) / len(self.episode_outcomes)

    @property
    def mean_turns(self) -> float:
        if not self.episode_turns:
            return 0.0
        return sum(self.episode_turns) / len(self.episode_turns)

    @property
    def total_turns(self) -> int:
        return sum(self.episode_turns)

    def get_agent_episodes(self, agent_name: str) -> List[int]:
        """Get episode indices played by a specific agent."""
        return [i for i, name in enumerate(self.agent_assignments) if name == agent_name]

    def get_agent_rewards(self, agent_name: str) -> List[float]:
        """Get rewards for episodes played by a specific agent."""
        return [self.episode_rewards[i] for i in self.get_agent_episodes(agent_name)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            # Core
            "episode_outcomes": [o.to_dict() for o in self.episode_outcomes],
            "conversation": self.conversation,

            # Identity
            "model_name": self.model_name,
            "benchmark_id": self.benchmark_id,
            "seed": self.seed,

            # Env config
            "env_name": self.env_name,
            "latent_id": self.latent_id,
            "prompt_id": self.prompt_id,
            "feedback_id": self.feedback_id,
            "reward_type": self.reward_type,
            "max_turns_per_episode": self.max_turns_per_episode,
            "env_params": self.env_params,

            # Agents
            "agent_assignments": self.agent_assignments,

            # Ground truth
            "episode_configs": self.episode_configs,

            # Reasoning trace (per assistant turn, None if no reasoning)
            "reasoning_trace": self.reasoning_trace,

            # Metadata
            "init_metadata": self.init_metadata,
            "final_metadata": self.final_metadata,
            "metadata": self.metadata,

            # Derived
            "episode_rewards": self.episode_rewards,
            "episode_turns": self.episode_turns,
            "cumulative_reward": self.cumulative_reward,
            "terminal_reward": self.terminal_reward,
            "initial_reward": self.initial_reward,
            "improvement": self.improvement,
            "mean_reward": self.mean_reward,
            "success_rate": self.success_rate,
            "mean_turns": self.mean_turns,
            "total_turns": self.total_turns,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TrajectoryResult:
        outcomes = [EpisodeOutcome.from_dict(o) for o in d.get("episode_outcomes", [])]
        return cls(
            episode_outcomes=outcomes,
            conversation=d.get("conversation", []),
            model_name=d.get("model_name", ""),
            benchmark_id=d.get("benchmark_id", ""),
            seed=d.get("seed", 0),
            env_name=d.get("env_name", ""),
            latent_id=d.get("latent_id", ""),
            prompt_id=d.get("prompt_id", ""),
            feedback_id=d.get("feedback_id", ""),
            reward_type=d.get("reward_type", ""),
            max_turns_per_episode=d.get("max_turns_per_episode", 0),
            env_params=d.get("env_params", {}),
            agent_assignments=d.get("agent_assignments", []),
            episode_configs=d.get("episode_configs", []),
            reasoning_trace=d.get("reasoning_trace", []),
            init_metadata=d.get("init_metadata", {}),
            final_metadata=d.get("final_metadata", {}),
            metadata=d.get("metadata", {}),
        )
