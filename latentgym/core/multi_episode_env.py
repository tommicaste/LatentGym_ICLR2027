"""
MultiEpisodeEnv — Base class for multi-episode benchmark environments.

Inherits from BaseTextEnv (skyrl-gym) for SkyRL training compatibility.
Receives pre-resolved episode_configs and composes them with prompt/feedback/reward.

The env does NOT resolve latents — that's done by make_env() in registry.py.
It just plays through episodes using the given configs.

Two levels:
    Level 1 (Homogeneous): Same SingleEpisodeEnv for all episodes (primary).
    Level 2 (Heterogeneous): Different SingleEpisodeEnv per episode via EpisodeDef.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

from .single_episode_env import SingleEpisodeEnv
from .prompt import PromptTemplate
from .feedback import FeedbackHandler
from .reward import RewardAggregator, RewardType
from .env_config import EpisodeDef


class MultiEpisodeEnv(BaseTextEnv):
    """Multi-episode environment that manages episode lifecycle.

    Receives pre-resolved episode_configs (from make_env or trajectory files).
    Handles episode transitions, prompt composition, feedback formatting,
    and reward aggregation.
    """

    def __init__(self, env_config: Optional[DictConfig] = None, extras: Dict[str, Any] = {}):
        """SkyRL-compatible constructor.

        Called in two contexts:

        Path 1 — Via from_configs() / make_env() (our eval pipeline):
            extras["extra_info"] has pre-resolved objects (core_env, prompt_template, etc.)

        Path 2 — Via skyrl_gym.make() (SkyRL training/eval pipeline):
            extras["extra_info"] has string IDs + trajectory_path.
            Components are resolved from the benchmark registry.
            Parquet extra_info example:
                {"trajectory_path": "/path/traj.json", "env_name": "bandits",
                 "prompt_id": "full_info", "feedback_id": "standard",
                 "reward_type": "cumulative", "latent_id": "loyal_favorite_0"}

        For direct use, prefer make_env() in registry.py.
        """
        super().__init__()

        if hasattr(self, "_initialized") and self._initialized:
            return

        extra_info = extras.get("extra_info", {})

        if extra_info.get("trajectory_path") and extra_info.get("env_name"):
            # Path 2: SkyRL called us with string IDs + trajectory file
            self._init_from_skyrl_extras(extra_info)
        elif extra_info.get("core_env"):
            # Path 1: from_configs/make_env passed pre-resolved objects
            self._core_env = extra_info.get("core_env")
            self._episode_defs = extra_info.get("episode_defs")
            self._episode_configs = extra_info.get("episode_configs", [])
            self._prompt_template = extra_info.get("prompt_template")
            self._feedback_handler = extra_info.get("feedback_handler")
            self._reward_aggregator = extra_info.get("reward_aggregator")
            self._num_episodes = extra_info.get("num_episodes", len(self._episode_configs))
            self._env_params = extra_info.get("env_params", {})
            self._metadata = extra_info.get("metadata", {})
        else:
            # Empty init (from_configs/from_episode_defs will set attrs later)
            self._core_env = None
            self._episode_defs = None
            self._episode_configs = []
            self._prompt_template = None
            self._feedback_handler = None
            self._reward_aggregator = None
            self._num_episodes = 0
            self._env_params = {}
            self._metadata = {}

        self._init_state()

    def _init_from_skyrl_extras(self, extra_info: Dict[str, Any]):
        """Initialize from SkyRL extras: resolve string IDs from registry + load trajectory.

        This is the path used when SkyRL calls skyrl_gym.make() with our registered env.
        The parquet extra_info contains string IDs (env_name, prompt_id, feedback_id)
        which we resolve to actual objects from the benchmark registry.
        Episode configs are loaded from the trajectory JSON file.
        """
        import json
        from .registry import _ENV_REGISTRY, _PROMPT_REGISTRY, _FEEDBACK_REGISTRY

        env_name = extra_info["env_name"]
        trajectory_path = extra_info["trajectory_path"]
        prompt_id = extra_info.get("prompt_id", "no_info")
        feedback_id = extra_info.get("feedback_id", "standard")
        reward_type_str = extra_info.get("reward_type", "cumulative")
        latent_id = extra_info.get("latent_id", "")

        # Load episode configs from trajectory JSON
        with open(trajectory_path, "r") as f:
            traj_data = json.load(f)
        episode_configs = traj_data.get("episodes", [])

        # Resolve core_env from registry (EnvRegistration has env_class + default params)
        env_reg = _ENV_REGISTRY.get(env_name)
        if env_reg is None:
            raise ValueError(
                f"Environment '{env_name}' not in benchmark registry. "
                f"Available: {list(_ENV_REGISTRY.keys())}. "
                f"Did you import the env package? (e.g., import latentgym.envs.bandits)"
            )

        # Resolve prompt template from registry
        prompt_cls = _PROMPT_REGISTRY.get(env_name, {}).get(prompt_id)
        if prompt_cls is None:
            available = list(_PROMPT_REGISTRY.get(env_name, {}).keys())
            raise ValueError(
                f"Prompt '{prompt_id}' not found for '{env_name}'. Available: {available}"
            )

        # Resolve feedback handler from registry
        feedback_cls = _FEEDBACK_REGISTRY.get(env_name, {}).get(feedback_id)
        if feedback_cls is None:
            available = list(_FEEDBACK_REGISTRY.get(env_name, {}).keys())
            raise ValueError(
                f"Feedback '{feedback_id}' not found for '{env_name}'. Available: {available}"
            )

        # env_params: used by prompt templates for system prompt construction.
        # Priority: trajectory metadata > SkyRL extras > env registration defaults.
        env_params = {**env_reg.env_params}
        env_params.update(extra_info.get("env_params", {}))
        traj_metadata = traj_data.get("metadata", {})
        env_params.update(traj_metadata.get("env_params", {}))

        # Create core_env with no args — it owns its own defaults
        env_cls = env_reg.env_class
        self._core_env = env_cls()
        self._episode_defs = None
        self._episode_configs = episode_configs
        self._prompt_template = prompt_cls()
        self._feedback_handler = feedback_cls()
        self._reward_aggregator = RewardAggregator(RewardType(reward_type_str))
        self._num_episodes = extra_info.get("num_episodes", len(episode_configs))
        env_params["latent_id"] = latent_id
        self._env_params = env_params
        self._metadata = {"latent_id": latent_id, "latent_mode": "trajectory"}

    def _init_state(self):
        """Initialize mutable state for a trajectory."""
        self._current_episode: int = 0
        self._episode_turn: int = 0
        self._total_turns: int = 0
        self._episode_rewards: List[float] = []
        self._turns_per_episode: List[int] = []
        self._episode_cumulative_reward: float = 0.0
        self._current_episode_info: Dict[str, Any] = {}
        self._initialized = True

    @classmethod
    def from_configs(
        cls,
        core_env: SingleEpisodeEnv,
        episode_configs: List[Dict[str, Any]],
        prompt_template: PromptTemplate,
        feedback_handler: FeedbackHandler,
        reward_aggregator: RewardAggregator,
        env_params: Dict[str, Any] = {},
        metadata: Dict[str, Any] = {},
    ) -> MultiEpisodeEnv:
        """Primary constructor: homogeneous env with pre-resolved episode_configs.

        Args:
            core_env: SingleEpisodeEnv instance used for all episodes
            episode_configs: Pre-resolved configs, one per episode
            prompt_template: For system prompt and episode transitions
            feedback_handler: For formatting observations
            reward_aggregator: Reward aggregation strategy
            env_params: Environment-level parameters (buttons, word_length, etc.)
            metadata: Extra metadata (latent_id, latent_mode, etc.)

        Returns:
            Configured MultiEpisodeEnv.
        """
        instance = cls.__new__(cls)
        BaseTextEnv.__init__(instance)

        instance._core_env = core_env
        instance._episode_defs = None
        instance._episode_configs = episode_configs
        instance._prompt_template = prompt_template
        instance._feedback_handler = feedback_handler
        instance._reward_aggregator = reward_aggregator
        instance._num_episodes = len(episode_configs)
        instance._env_params = env_params
        instance._metadata = metadata
        instance._init_state()
        return instance

    @classmethod
    def from_episode_defs(
        cls,
        episode_defs: List[EpisodeDef],
        prompt_template: PromptTemplate,
        feedback_handler: FeedbackHandler,
        reward_aggregator: RewardAggregator,
        env_params: Dict[str, Any] = {},
        metadata: Dict[str, Any] = {},
    ) -> MultiEpisodeEnv:
        """Heterogeneous constructor: different core_env per episode.

        Args:
            episode_defs: List of EpisodeDef, one per episode
            prompt_template: Default prompt template
            feedback_handler: Default feedback handler
            reward_aggregator: Reward aggregation strategy
            env_params: Environment-level parameters
            metadata: Extra metadata

        Returns:
            Configured MultiEpisodeEnv.
        """
        instance = cls.__new__(cls)
        BaseTextEnv.__init__(instance)

        instance._core_env = None
        instance._episode_defs = episode_defs
        instance._episode_configs = [ed.episode_config for ed in episode_defs]
        instance._prompt_template = prompt_template
        instance._feedback_handler = feedback_handler
        instance._reward_aggregator = reward_aggregator
        instance._num_episodes = len(episode_defs)
        instance._env_params = env_params
        instance._metadata = metadata
        instance._init_state()
        return instance

    # ========================================================================
    # BaseTextEnv interface
    # ========================================================================

    def init(self, prompt: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """Initialize the multi-episode trajectory.

        Approach C (hybrid prompt handling):
        - If prompt is empty or placeholder → construct our own via layered composition
        - If prompt has real content → use it, append episode header + initial obs
          (backwards compat with old-style parquets that had full prompts baked in)

        In both cases, the episode header and initial observation are appended.
        """
        self._init_state()

        core_env = self._get_core_env(0)
        episode_config = self._episode_configs[0] if self._episode_configs else {}

        # Reset first episode to get initial observation
        initial_obs = core_env.reset(episode_config)
        episode_header = f"\n\n--- Episode 1 of {self._num_episodes} ---\n"

        # Determine system prompt
        provided_prompt = self._extract_system_prompt(prompt)

        if provided_prompt:
            # Backwards compat: use provided prompt, append episode info
            system_content = provided_prompt + episode_header + initial_obs
        else:
            # Standard path: construct via layered composition
            game_rules = core_env.get_game_rules()
            system_prompt = self._prompt_template.initial_system_prompt(
                game_rules=game_rules,
                env_params=self._env_params,
                num_episodes=self._num_episodes,
            )
            system_content = system_prompt + episode_header + initial_obs

        conversation = [{"role": "system", "content": system_content}]

        metadata = {
            "env_name": self.__class__.__name__,
            "num_episodes": self._num_episodes,
            "env_params": self._env_params,
            "episode_configs": self._episode_configs,
            **self._metadata,
        }

        # Include optional fields if available
        if self._reward_aggregator:
            metadata["reward_type"] = self._reward_aggregator.reward_type.value
        max_turns = self._env_params.get("max_turns_per_episode", 0)
        if max_turns:
            metadata["max_turns_per_episode"] = max_turns

        return conversation, metadata

    @staticmethod
    def _extract_system_prompt(prompt: List[Dict[str, str]]) -> str:
        """Extract system prompt content from a prompt list.

        Returns empty string if prompt is empty, has no system message,
        or the system message content is empty (placeholder).
        """
        if not prompt:
            return ""
        for msg in prompt:
            if msg.get("role") == "system":
                content = msg.get("content", "").strip()
                return content
        return ""

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """Execute one action in the current episode."""
        self._episode_turn += 1
        self._total_turns += 1

        core_env = self._get_core_env(self._current_episode)

        # Delegate to core env
        raw_feedback, step_reward, episode_done, info = core_env.step(action)
        self._current_episode_info = info

        # Format intra-episode feedback (uses per-episode handler for heterogeneous)
        feedback_handler = self._get_feedback_handler(self._current_episode)
        formatted_feedback = feedback_handler.format_step_feedback(
            raw_feedback, self._current_episode, self._episode_turn, info
        )

        episode_just_ended = episode_done

        if episode_done:
            # Record episode results
            episode_reward = step_reward
            self._episode_rewards.append(episode_reward)
            self._turns_per_episode.append(self._episode_turn)
            self._episode_cumulative_reward += episode_reward

            if self._current_episode < self._num_episodes - 1:
                # Non-last episode: end_feedback + transition_msg + next episode
                end_feedback = feedback_handler.format_episode_end_feedback(
                    self._current_episode, episode_reward, info
                )
                prompt_tpl = self._get_prompt_template(self._current_episode)
                transition_msg = prompt_tpl.episode_transition_message(
                    self._current_episode,
                    self._num_episodes,
                    episode_reward,
                    info,
                )

                self._current_episode += 1
                self._episode_turn = 0

                # Reset next episode
                next_core_env = self._get_core_env(self._current_episode)
                next_config = (
                    self._episode_configs[self._current_episode]
                    if self._current_episode < len(self._episode_configs)
                    else {}
                )
                next_obs = next_core_env.reset(next_config)

                # For heterogeneous: if next episode has different game rules,
                # include them in the transition
                game_rules_reminder = ""
                if self._episode_defs is not None:
                    next_rules = next_core_env.get_game_rules()
                    game_rules_reminder = f"\n{next_rules}\n"

                formatted_feedback = (
                    f"{formatted_feedback}\n\n"
                    f"{end_feedback}\n\n"
                    f"{transition_msg}{game_rules_reminder}\n"
                    f"{next_obs}"
                )
            else:
                # Last episode: end_feedback only (no transition)
                end_feedback = feedback_handler.format_episode_end_feedback(
                    self._current_episode, episode_reward, info
                )
                formatted_feedback = f"{formatted_feedback}\n\n{end_feedback}"

        # Trajectory done when all episodes completed
        trajectory_done = False
        if episode_done and len(self._episode_rewards) >= self._num_episodes:
            trajectory_done = True

        # Compute reward via aggregator
        reward = self._reward_aggregator.compute_step_reward(
            self._episode_rewards, episode_just_ended, trajectory_done
        )

        step_metadata = {
            "turn": self._total_turns,
            "episode": self._current_episode,
            "episode_turn": self._episode_turn,
            "total_episodes": self._num_episodes,
            "episode_rewards": self._episode_rewards.copy(),
            "turns_per_episode": self._turns_per_episode.copy(),
            "cumulative_reward": self._episode_cumulative_reward,
        }

        # Include optional fields if available
        if self._metadata.get("latent_id"):
            step_metadata["latent_id"] = self._metadata["latent_id"]
        if self._reward_aggregator:
            step_metadata["reward_type"] = self._reward_aggregator.reward_type.value
        max_turns = self._env_params.get("max_turns_per_episode", 0)
        if max_turns:
            step_metadata["max_turns_per_episode"] = max_turns

        return {
            "observations": [{"role": "user", "content": formatted_feedback}],
            "reward": reward,
            "done": trajectory_done,
            "metadata": step_metadata,
        }

    def close(self):
        """Clean up all core environments."""
        if self._core_env is not None:
            self._core_env.close()
        if self._episode_defs:
            for ed in self._episode_defs:
                env = ed.core_env
                if hasattr(env, "close"):
                    env.close()

    # ========================================================================
    # Internal helpers
    # ========================================================================

    def _get_core_env(self, episode_idx: int) -> SingleEpisodeEnv:
        """Get the core env for a given episode.
        Homogeneous: always returns self._core_env.
        Heterogeneous: returns per-episode core_env from EpisodeDef."""
        if self._episode_defs is not None:
            ed = self._episode_defs[episode_idx]
            env = ed.core_env
            if callable(env) and not isinstance(env, SingleEpisodeEnv):
                env = env()  # Call factory
            return env
        return self._core_env

    def _get_prompt_template(self, episode_idx: int) -> PromptTemplate:
        """Get prompt template for a given episode.
        Homogeneous: always returns self._prompt_template.
        Heterogeneous: returns per-episode override if set."""
        if self._episode_defs and self._episode_defs[episode_idx].prompt_template:
            return self._episode_defs[episode_idx].prompt_template
        return self._prompt_template

    def _get_feedback_handler(self, episode_idx: int) -> FeedbackHandler:
        """Get feedback handler for a given episode.
        Homogeneous: always returns self._feedback_handler.
        Heterogeneous: returns per-episode override if set."""
        if self._episode_defs and self._episode_defs[episode_idx].feedback_handler:
            return self._episode_defs[episode_idx].feedback_handler
        return self._feedback_handler

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def num_episodes(self) -> int:
        return self._num_episodes

    @property
    def episode_rewards(self) -> List[float]:
        return self._episode_rewards.copy()

    @property
    def current_episode(self) -> int:
        return self._current_episode

    def get_trajectory_info(self) -> Dict[str, Any]:
        return {
            "num_episodes": self._num_episodes,
            "env_params": self._env_params,
            "episode_rewards": self._episode_rewards,
            "turns_per_episode": self._turns_per_episode,
            **self._metadata,
        }
