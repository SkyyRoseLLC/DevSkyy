"""
DevSkyy Enterprise Agent Routing System
MCP-optimized agent configuration loading and task routing

Version: 1.0.0
Truth Protocol Compliance: CLAUDE.md
"""

from agents.loader import (
    AgentConfigLoader,
    AgentConfigError,
    ConfigValidationError
)

from agents.router import (
    AgentRouter,
    TaskRequest,
    TaskResult,
    TaskType
)

__version__ = "1.0.0"
__all__ = [
    "AgentConfigLoader",
    "AgentConfigError",
    "ConfigValidationError",
    "AgentRouter",
    "TaskRequest",
    "TaskResult",
    "TaskType",
]
