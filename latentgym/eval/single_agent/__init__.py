"""
Single-agent evaluation module.

Two runner backends:
- LocalRunner: Uses SkyRL's SkyRLGymGenerator for local vLLM/SGLang models
- APIRunner: Uses our lightweight runner for API models (OpenAI, Anthropic, etc.)
"""
from .api_runner import APIRunner
from .metrics import (
    compute_single_agent_metrics,
    compute_comparison_metrics,
    compute_detailed_metrics,
    compute_per_agent_metrics,
    format_metrics_summary,
)
