"""
Agent scheduling for multi-agent evaluation.

Defines which model plays which episodes in a trajectory.
Supports single-agent, two-agent switch, and multi-agent schedules.

Usage:
    # Single agent
    schedule = single_agent_schedule(model_a)

    # Two agents with switch at episode 5
    schedule = two_agent_schedule(model_a, model_b, switch_at=5)

    # All four comparison schedules (P_f, Q_b, P_f→Q_b, Q_b→P_f)
    schedules = create_comparison_schedules(finetuned, base, switch_at=5)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from latentgym.eval.model_interface import ModelInterface


@dataclass
class AgentConfig:
    """Configuration for a single agent.

    Attributes:
        name: Unique identifier (e.g., "finetuned", "base", "gpt-4o")
        model: The ModelInterface to use for generation
        sampling_params: Generation parameters (temperature, max_tokens, etc.)
        metadata: Optional additional metadata
    """
    name: str
    model: ModelInterface
    sampling_params: Dict[str, Any] = field(default_factory=lambda: {
        "temperature": 0.7,
        "max_tokens": 512,
    })
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            raise ValueError("Agent name cannot be empty")


@dataclass
class AgentSchedule:
    """Defines which agent plays which episodes in a trajectory.

    Supports:
        - Single agent:     AgentSchedule([agent], [])
        - Two-agent switch: AgentSchedule([agent1, agent2], [5])
        - Multi-switch:     AgentSchedule([a1, a2, a3], [3, 7])

    Attributes:
        agents: List of agent configurations in order
        switch_at_episodes: Episode indices where switches occur
            - [] means single agent throughout
            - [5] means agents[0] plays eps 0-4, agents[1] plays 5+
            - [3, 7] means agents[0] plays 0-2, agents[1] plays 3-6, agents[2] plays 7+
    """
    agents: List[AgentConfig]
    switch_at_episodes: List[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.agents:
            raise ValueError("At least one agent must be provided")
        if len(self.switch_at_episodes) >= len(self.agents):
            raise ValueError(
                f"Number of switch points ({len(self.switch_at_episodes)}) "
                f"must be less than number of agents ({len(self.agents)})"
            )
        if self.switch_at_episodes:
            if sorted(self.switch_at_episodes) != self.switch_at_episodes:
                raise ValueError("Switch points must be in ascending order")
            if any(s < 0 for s in self.switch_at_episodes):
                raise ValueError("Switch points must be non-negative")

    @property
    def is_single_agent(self) -> bool:
        return len(self.agents) == 1

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    @property
    def agent_names(self) -> List[str]:
        return [a.name for a in self.agents]

    def get_agent_for_episode(self, episode_idx: int) -> AgentConfig:
        """Get the agent that should play a specific episode."""
        if episode_idx < 0:
            raise ValueError(f"Episode index must be non-negative, got {episode_idx}")

        agent_idx = 0
        for switch_point in self.switch_at_episodes:
            if episode_idx >= switch_point:
                agent_idx += 1
            else:
                break

        agent_idx = min(agent_idx, len(self.agents) - 1)
        return self.agents[agent_idx]

    def get_agent_assignments(self, num_episodes: int) -> List[str]:
        """Get agent name for each episode."""
        return [self.get_agent_for_episode(i).name for i in range(num_episodes)]

    def get_episodes_per_agent(self, num_episodes: int) -> Dict[str, List[int]]:
        """Get which episodes each agent plays."""
        result = {agent.name: [] for agent in self.agents}
        for i in range(num_episodes):
            agent = self.get_agent_for_episode(i)
            result[agent.name].append(i)
        return result

    def describe(self, num_episodes: Optional[int] = None) -> str:
        """Get human-readable description of this schedule."""
        if self.is_single_agent:
            return f"Single agent: {self.agents[0].name}"

        parts = []
        prev_switch = 0
        for i, switch in enumerate(self.switch_at_episodes):
            parts.append(f"{self.agents[i].name} (eps {prev_switch}-{switch - 1})")
            prev_switch = switch

        if num_episodes:
            parts.append(f"{self.agents[-1].name} (eps {prev_switch}-{num_episodes - 1})")
        else:
            parts.append(f"{self.agents[-1].name} (eps {prev_switch}+)")

        return " → ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agents": [a.name for a in self.agents],
            "switch_at_episodes": self.switch_at_episodes,
            "is_single_agent": self.is_single_agent,
        }


# ============================================================================
# Factory functions
# ============================================================================

def single_agent_schedule(agent: AgentConfig) -> AgentSchedule:
    """Create a schedule with a single agent for all episodes."""
    return AgentSchedule(agents=[agent], switch_at_episodes=[])


def two_agent_schedule(
    first_agent: AgentConfig,
    second_agent: AgentConfig,
    switch_at: int,
) -> AgentSchedule:
    """Create a schedule that switches between two agents at a specific episode."""
    return AgentSchedule(
        agents=[first_agent, second_agent],
        switch_at_episodes=[switch_at],
    )


def create_comparison_schedules(
    finetuned: AgentConfig,
    base: AgentConfig,
    switch_at: int,
) -> Dict[str, AgentSchedule]:
    """Create all four comparison schedules for evaluation.

    Returns:
        Dict with keys:
            "P_f":      Finetuned only (all episodes)
            "Q_b":      Base only (all episodes)
            "P_f->Q_b": Finetuned then base (switch at switch_at)
            "Q_b->P_f": Base then finetuned (switch at switch_at)
    """
    return {
        "P_f": single_agent_schedule(finetuned),
        "Q_b": single_agent_schedule(base),
        "P_f->Q_b": two_agent_schedule(finetuned, base, switch_at),
        "Q_b->P_f": two_agent_schedule(base, finetuned, switch_at),
    }
