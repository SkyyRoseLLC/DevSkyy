"""
DevSkyy Agent Orchestration System
Programmatic agent configuration loading and task routing

Truth Protocol Compliance: CLAUDE_20-10_MASTER.md
"""

from agents.loader import (
    AgentConfigLoader,
    AgentConfiguration,
    AgentType,
    AgentStatus,
    OrchestrationCommand,
    PerformanceSLO
)

from agents.router import (
    AgentRouter,
    TaskType,
    TaskContext,
    RoutingDecision
)

__version__ = "2.0.0"
__all__ = [
    # Loader
    "AgentConfigLoader",
    "AgentConfiguration",
    "AgentType",
    "AgentStatus",
    "OrchestrationCommand",
    "PerformanceSLO",
    # Router
    "AgentRouter",
    "TaskType",
    "TaskContext",
    "RoutingDecision",
]
