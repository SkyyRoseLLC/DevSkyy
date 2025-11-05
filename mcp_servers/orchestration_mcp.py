#!/usr/bin/env python3
"""
DevSkyy Orchestration MCP Server
System monitoring, workflow automation, and health management via MCP

Port: 5006
Category: Orchestration
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_servers.shared.mcp_base import BaseMCPServer, load_env_config
from fastmcp import Context
from typing import Any, Dict

# Load configuration
config = load_env_config(category="orchestration", port=5006)


class OrchestrationMCPServer(BaseMCPServer):
    """System orchestration and monitoring MCP server"""

    def __init__(self, config):
        super().__init__(config)
        self.register_tools()

    def register_tools(self):
        """Register orchestration-specific MCP tools"""

        @self.mcp.tool()
        async def get_system_health(ctx: Context, detailed: bool = False) -> str:
            """
            Get comprehensive system health status

            Args:
                detailed: Include detailed component metrics

            Returns:
                System health report with all services and agents
            """
            result = await self.client.request(
                method="GET",
                endpoint="/api/v1/monitoring/health",
                params={"detailed": str(detailed).lower()}
            )

            return self.format_response(result, "System Health")

        @self.mcp.tool()
        async def list_all_agents(ctx: Context, category: str = "all") -> str:
            """
            List all available AI agents in the system

            Args:
                category: Filter by category (all, backend, frontend, ml, content)

            Returns:
                List of agents with status and capabilities
            """
            result = await self.client.request(
                method="GET",
                endpoint="/api/v1/agents",
                params={"category": category}
            )

            return self.format_response(result, "DevSkyy AI Agents")

        @self.mcp.tool()
        async def execute_workflow(
            ctx: Context,
            workflow_name: str,
            parameters: str = "{}"
        ) -> str:
            """
            Execute predefined automation workflow

            Args:
                workflow_name: Name of workflow to execute
                parameters: JSON string of workflow parameters

            Returns:
                Workflow execution status and results
            """
            import json

            try:
                workflow_params = json.loads(parameters)
            except json.JSONDecodeError:
                return "❌ Error: Invalid JSON format for parameters"

            result = await self.client.request(
                method="POST",
                endpoint="/api/v1/workflows/execute",
                data={
                    "workflow_name": workflow_name,
                    "parameters": workflow_params
                }
            )

            return self.format_response(result, f"Workflow: {workflow_name}")

        @self.mcp.tool()
        async def get_system_metrics(
            ctx: Context,
            metric_type: str = "all",
            time_range: str = "1h"
        ) -> str:
            """
            Get system performance metrics

            Args:
                metric_type: Metric type (all, cpu, memory, requests, errors)
                time_range: Time range (1h, 24h, 7d, 30d)

            Returns:
                Performance metrics and trends
            """
            result = await self.client.request(
                method="GET",
                endpoint="/api/v1/monitoring/metrics",
                params={
                    "metric_type": metric_type,
                    "time_range": time_range
                }
            )

            return self.format_response(result, "System Metrics")

        @self.mcp.tool()
        async def trigger_self_healing(
            ctx: Context,
            issue_type: str,
            severity: str = "high"
        ) -> str:
            """
            Trigger self-healing process for detected issues

            Args:
                issue_type: Type of issue (syntax_error, security_vulnerability, performance)
                severity: Issue severity (low, medium, high, critical)

            Returns:
                Self-healing execution results
            """
            result = await self.client.request(
                method="POST",
                endpoint="/api/v1/healing/trigger",
                data={
                    "issue_type": issue_type,
                    "severity": severity
                }
            )

            return self.format_response(result, "Self-Healing Process")

        @self.mcp.tool()
        async def list_workflows(ctx: Context, status: str = "all") -> str:
            """
            List available automation workflows

            Args:
                status: Filter by status (all, active, scheduled, completed)

            Returns:
                List of workflows with descriptions and schedules
            """
            result = await self.client.request(
                method="GET",
                endpoint="/api/v1/workflows",
                params={"status": status}
            )

            return self.format_response(result, "Automation Workflows")

        @self.mcp.tool()
        async def get_error_logs(
            ctx: Context,
            severity: str = "all",
            limit: int = 50
        ) -> str:
            """
            Retrieve system error logs

            Args:
                severity: Filter by severity (all, error, warning, critical)
                limit: Maximum number of logs to return

            Returns:
                Recent error logs with timestamps and details
            """
            result = await self.client.request(
                method="GET",
                endpoint="/api/v1/monitoring/logs",
                params={
                    "severity": severity,
                    "limit": limit
                }
            )

            return self.format_response(result, "Error Logs")

        @self.mcp.tool()
        async def schedule_task(
            ctx: Context,
            task_name: str,
            schedule: str,
            agent_name: str,
            parameters: str = "{}"
        ) -> str:
            """
            Schedule recurring task execution

            Args:
                task_name: Name of task to schedule
                schedule: Cron schedule expression
                agent_name: Agent to execute task
                parameters: JSON string of task parameters

            Returns:
                Scheduled task details and next execution time
            """
            import json

            try:
                task_params = json.loads(parameters)
            except json.JSONDecodeError:
                return "❌ Error: Invalid JSON format for parameters"

            result = await self.client.request(
                method="POST",
                endpoint="/api/v1/scheduler/schedule",
                data={
                    "task_name": task_name,
                    "schedule": schedule,
                    "agent_name": agent_name,
                    "parameters": task_params
                }
            )

            return self.format_response(result, f"Scheduled Task: {task_name}")

        @self.mcp.tool()
        async def get_api_usage_stats(
            ctx: Context,
            time_range: str = "24h"
        ) -> str:
            """
            Get API usage statistics

            Args:
                time_range: Time range (1h, 24h, 7d, 30d)

            Returns:
                API usage stats with endpoint breakdown
            """
            result = await self.client.request(
                method="GET",
                endpoint="/api/v1/monitoring/api-stats",
                params={"time_range": time_range}
            )

            return self.format_response(result, "API Usage Statistics")

        @self.mcp.tool()
        async def backup_system_state(ctx: Context, backup_type: str = "full") -> str:
            """
            Create system backup

            Args:
                backup_type: Backup type (full, config, database)

            Returns:
                Backup details and storage location
            """
            result = await self.client.request(
                method="POST",
                endpoint="/api/v1/orchestration/backup",
                data={"backup_type": backup_type}
            )

            return self.format_response(result, "System Backup")


# Initialize and run server
if __name__ == "__main__":
    server = OrchestrationMCPServer(config)
    server.run()
