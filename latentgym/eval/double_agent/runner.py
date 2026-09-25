"""
ScheduledRunner — Run trajectories with agent scheduling (single or multi-agent).

Supports any AgentSchedule: single agent, two-agent switch, multi-switch.
This replaces both the old APIRunner.run_trajectory (single agent) and
DoubleAgentRunner (two agents) with one unified runner.

For single-agent eval, use a single_agent_schedule.
For double-agent eval, use a two_agent_schedule.
For complex schedules, construct AgentSchedule directly.

Current setup for local models (double-agent):
    Requires two separate vLLM servers running on different GPUs/ports:
        GPU 0: vllm serve /path/to/finetuned --port 8000
        GPU 1: vllm serve /path/to/base --port 8001
    Then create two VLLMModel instances pointing to each server.

Future: SequentialModelLoader
    For single-GPU setups where both models can't fit in memory simultaneously,
    a SequentialModelLoader could be added. It would:
    1. Load model A into GPU memory
    2. Run all episodes assigned to model A
    3. Unload model A, free GPU memory
    4. Load model B into GPU memory
    5. Run all episodes assigned to model B
    The conversation (text) carries over — only the generating model changes.
    This works because agent switches happen at episode boundaries, so we can
    batch all of one agent's episodes before switching.
    This differs from the current setup where both models must be available
    simultaneously (as separate servers or API endpoints).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from latentgym.core.multi_episode_env import MultiEpisodeEnv
from latentgym.eval.types import EpisodeOutcome, TrajectoryResult, OutcomeType
from .agent_scheduler import AgentSchedule

logger = logging.getLogger(__name__)


class ScheduledRunner:
    """Run a trajectory with any agent schedule.

    The schedule determines which model generates at each episode.
    Conversation context carries over across agent switches —
    the new agent sees the full history from previous agents.

    Captures all data: per-step actions/feedback, per-episode outcomes,
    full conversation, and all metadata.
    """

    def __init__(self, schedule: AgentSchedule):
        self.schedule = schedule

    async def run_trajectory(
        self,
        env: MultiEpisodeEnv,
        seed: int = 0,
        max_total_turns: int = 500,
        success_threshold: float = 0.01,
    ) -> TrajectoryResult:
        """Run a single trajectory using the agent schedule."""
        conversation, init_metadata = env.init([])

        # Extract env config from init metadata
        env_name = init_metadata.get("env_name", "")
        latent_id = init_metadata.get("latent_id", "")
        reward_type = init_metadata.get("reward_type", "")
        max_turns_per_ep = init_metadata.get("max_turns_per_episode", 0)
        env_params = init_metadata.get("env_params", {})
        episode_configs = init_metadata.get("episode_configs", [])

        episode_outcomes: List[EpisodeOutcome] = []
        reasoning_trace: List[Optional[str]] = []
        current_episode = 0
        episode_turn = 0
        final_metadata: Dict[str, Any] = {}

        done = False
        while not done and (episode_turn + sum(o.turns for o in episode_outcomes)) < max_total_turns:
            # Get active agent for current episode
            agent = self.schedule.get_agent_for_episode(current_episode)

            response = await agent.model.generate(conversation, **agent.sampling_params)
            episode_turn += 1

            # Record reasoning separately (NOT fed back into conversation)
            reasoning_trace.append(response.reasoning)

            # Only action text goes into conversation context
            conversation.append({"role": "assistant", "content": response.text})

            step_result = env.step(response.text)
            obs = step_result["observations"]
            reward = step_result["reward"]
            done = step_result["done"]
            step_metadata = step_result["metadata"]

            if obs:
                conversation.extend(obs)

            # Track episode transitions
            new_episode = step_metadata.get("episode", 0)
            ep_rewards = step_metadata.get("episode_rewards", [])
            ep_turns = step_metadata.get("turns_per_episode", [])

            if new_episode != current_episode or done:
                # Record outcomes for newly completed episodes
                while len(episode_outcomes) < len(ep_rewards):
                    idx = len(episode_outcomes)
                    ep_agent = self.schedule.get_agent_for_episode(idx)
                    ep_reward = ep_rewards[idx]
                    ep_turn_count = ep_turns[idx] if idx < len(ep_turns) else episode_turn

                    # Determine outcome type
                    if ep_reward >= success_threshold:
                        outcome_type = OutcomeType.WIN
                    elif ep_turn_count >= max_turns_per_ep > 0:
                        outcome_type = OutcomeType.TIMEOUT
                    elif ep_reward > 0:
                        outcome_type = OutcomeType.PARTIAL
                    else:
                        outcome_type = OutcomeType.LOSS

                    gt = episode_configs[idx] if idx < len(episode_configs) else {}

                    episode_outcomes.append(EpisodeOutcome(
                        episode_idx=idx,
                        reward=ep_reward,
                        turns=ep_turn_count,
                        success=ep_reward >= success_threshold,
                        agent_name=ep_agent.name,
                        latent_id=step_metadata.get("latent_id", latent_id),
                        max_turns=max_turns_per_ep,
                        outcome_type=outcome_type,
                        ground_truth=gt,
                    ))

                current_episode = new_episode
                episode_turn = 0

            final_metadata = step_metadata

        env.close()

        # Build model name from schedule
        if self.schedule.is_single_agent:
            model_name = self.schedule.agents[0].name
        else:
            model_name = "→".join(
                f"{a.name}" for a in self.schedule.agents
            ) + f"@ep{'_'.join(str(s) for s in self.schedule.switch_at_episodes)}"

        # Agent assignments
        agent_assignments = [
            self.schedule.get_agent_for_episode(i).name
            for i in range(len(episode_outcomes))
        ]

        return TrajectoryResult(
            episode_outcomes=episode_outcomes,
            conversation=conversation,
            model_name=model_name,
            benchmark_id=init_metadata.get("benchmark_id", ""),
            seed=seed,
            env_name=env_name,
            latent_id=latent_id,
            prompt_id=init_metadata.get("prompt_id", ""),
            feedback_id=init_metadata.get("feedback_id", ""),
            reward_type=reward_type,
            max_turns_per_episode=max_turns_per_ep,
            env_params=env_params,
            agent_assignments=agent_assignments,
            episode_configs=episode_configs,
            reasoning_trace=reasoning_trace,
            init_metadata=init_metadata,
            final_metadata=final_metadata,
            metadata={"schedule": self.schedule.to_dict()},
        )


# Keep DoubleAgentRunner as a convenience alias
class DoubleAgentRunner:
    """Convenience wrapper for two-agent evaluation.

    Equivalent to ScheduledRunner with a two_agent_schedule.
    """

    def __init__(self, model_a, model_b):
        from .agent_scheduler import AgentConfig
        self.model_a = model_a
        self.model_b = model_b
        self._agent_a = AgentConfig(name=model_a.name, model=model_a)
        self._agent_b = AgentConfig(name=model_b.name, model=model_b)

    async def run_trajectory(
        self,
        env: MultiEpisodeEnv,
        switch_episode: int,
        seed: int = 0,
        max_total_turns: int = 500,
        success_threshold: float = 0.01,
    ) -> TrajectoryResult:
        from .agent_scheduler import two_agent_schedule
        schedule = two_agent_schedule(self._agent_a, self._agent_b, switch_episode)
        runner = ScheduledRunner(schedule)
        return await runner.run_trajectory(env, seed, max_total_turns, success_threshold)
