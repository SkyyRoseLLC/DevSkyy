"""
Base Agent Class
Foundation for all DevSkyy agents following enterprise patterns
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent execution status"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class AgentPriority(Enum):
    """Agent priority levels"""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


class BaseAgent(ABC):
    """
    Base class for all DevSkyy agents.

    Provides:
    - Lifecycle management (initialize, execute, cleanup)
    - Status tracking
    - Error handling
    - Logging
    - Metrics collection
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str = "",
        priority: AgentPriority = AgentPriority.NORMAL
    ):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.priority = priority
        self.status = AgentStatus.IDLE
        self.created_at = datetime.now()
        self.last_executed = None
        self.execution_count = 0
        self.error_count = 0
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")

    @abstractmethod
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's main task.

        Args:
            task: Task parameters

        Returns:
            Result dictionary
        """
        pass

    async def initialize(self) -> bool:
        """
        Initialize agent resources.

        Returns:
            True if successful
        """
        try:
            self.logger.info(f"Initializing agent: {self.name}")
            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False

    async def cleanup(self) -> bool:
        """
        Cleanup agent resources.

        Returns:
            True if successful
        """
        try:
            self.logger.info(f"Cleaning up agent: {self.name}")
            return True
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            return False

    async def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the complete agent lifecycle.

        Args:
            task: Task parameters

        Returns:
            Result dictionary with status
        """
        try:
            self.status = AgentStatus.RUNNING
            self.last_executed = datetime.now()
            self.execution_count += 1

            # Initialize if needed
            await self.initialize()

            # Execute main task
            result = await self.execute(task)

            # Cleanup
            await self.cleanup()

            self.status = AgentStatus.COMPLETED
            return {
                "status": "success",
                "agent_id": self.agent_id,
                "result": result,
                "executed_at": self.last_executed.isoformat()
            }

        except Exception as e:
            self.status = AgentStatus.FAILED
            self.error_count += 1
            self.logger.error(f"Agent execution failed: {e}")

            return {
                "status": "error",
                "agent_id": self.agent_id,
                "error": str(e),
                "executed_at": self.last_executed.isoformat() if self.last_executed else None
            }

    def get_status(self) -> Dict[str, Any]:
        """Get agent status information"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "status": self.status.value,
            "priority": self.priority.value,
            "execution_count": self.execution_count,
            "error_count": self.error_count,
            "last_executed": self.last_executed.isoformat() if self.last_executed else None,
            "uptime_seconds": (datetime.now() - self.created_at).total_seconds()
        }

    def pause(self):
        """Pause agent execution"""
        self.status = AgentStatus.PAUSED
        self.logger.info(f"Agent paused: {self.name}")

    def resume(self):
        """Resume agent execution"""
        self.status = AgentStatus.IDLE
        self.logger.info(f"Agent resumed: {self.name}")

    def reset(self):
        """Reset agent counters"""
        self.execution_count = 0
        self.error_count = 0
        self.last_executed = None
        self.status = AgentStatus.IDLE
        self.logger.info(f"Agent reset: {self.name}")
