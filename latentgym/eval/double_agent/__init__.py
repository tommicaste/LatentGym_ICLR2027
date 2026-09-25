"""Double-agent and multi-agent evaluation module."""
from .runner import ScheduledRunner, DoubleAgentRunner
from .metrics import compute_double_agent_metrics
from .agent_scheduler import (
    AgentConfig,
    AgentSchedule,
    single_agent_schedule,
    two_agent_schedule,
    create_comparison_schedules,
)
