#!/usr/bin/env python3
"""
DevSkyy Infrastructure MCP Server
Code scanning, fixing, security, and self-healing via MCP

Port: 5001
Category: Infrastructure
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_servers.shared.mcp_base import BaseMCPServer, load_env_config
from fastmcp import Context
from typing import Any, Dict
import json

# Load configuration
config = load_env_config(category="infrastructure", port=5001)


class InfrastructureMCPServer(BaseMCPServer):
    """Infrastructure automation MCP server"""

    def __init__(self, config):
        super().__init__(config)
        self.register_tools()

    def register_tools(self):
        """Register all infrastructure tools"""

        @self.mcp.tool()
        async def scan_code(
            ctx: Context,
            directory: str,
            include_security: bool = True,
            include_performance: bool = True,
            include_best_practices: bool = True
        ) -> str:
            """
            Scan code for errors, security issues, and performance bottlenecks.

            Args:
                directory: Path to directory to scan
                include_security: Include security vulnerability scanning
                include_performance: Include performance analysis
                include_best_practices: Include best practices checking

            Returns:
                Detailed scan results with errors, warnings, and suggestions
            """
            options = {
                "security": include_security,
                "performance": include_performance,
                "best_practices": include_best_practices,
            }

            result = await self.client.request(
                "POST",
                "/api/v1/agents/scanner/scan",
                data={"directory": directory, "options": options}
            )

            return self.format_response(result, f"Code Scan Results: {directory}")

        @self.mcp.tool()
        async def fix_code(ctx: Context, issues_json: str) -> str:
            """
            Automatically fix code issues identified by the scanner.

            Args:
                issues_json: JSON string of issues to fix (from scan results)

            Returns:
                Summary of fixes applied
            """
            try:
                issues = json.loads(issues_json)
            except json.JSONDecodeError:
                return "❌ Error: Invalid JSON format for issues"

            result = await self.client.request(
                "POST",
                "/api/v1/agents/code_fixer/fix",
                data={"issues": issues}
            )

            return self.format_response(result, "Code Fix Results")

        @self.mcp.tool()
        async def self_healing_check(ctx: Context) -> str:
            """
            Run self-healing system check and auto-repair.

            Monitors system health and automatically repairs issues:
            - Service health checks
            - Resource monitoring (CPU, memory, disk)
            - Auto-restart failed services
            - Clear caches and temporary files
            - Database optimization

            Returns:
                Health status and repairs performed
            """
            result = await self.client.request(
                "POST",
                "/api/v1/agents/self_healing/check"
            )

            return self.format_response(result, "Self-Healing System Check")

        @self.mcp.tool()
        async def security_scan(ctx: Context) -> str:
            """
            Comprehensive security vulnerability scan.

            Performs deep security analysis including:
            - SAST (Static Application Security Testing)
            - Dependency vulnerability scanning
            - Container security assessment
            - Authentication/authorization review
            - Data protection compliance check

            Returns detailed security report with remediation steps.
            """
            result = await self.client.request(
                "POST",
                "/api/v1/security/comprehensive-scan",
                data={"scan_type": "comprehensive", "include_remediation": True}
            )

            return self.format_response(result, "Security Vulnerability Scan")

        @self.mcp.tool()
        async def security_remediate(ctx: Context, issue_ids: str) -> str:
            """
            Automatically remediate security vulnerabilities.

            Args:
                issue_ids: Comma-separated list of security issue IDs to fix

            Applies automated fixes for:
            - Code injection vulnerabilities
            - Authentication bypasses
            - Insecure configurations
            - Dependency vulnerabilities
            - Container security issues

            Returns remediation report with applied fixes.
            """
            issue_list = [id.strip() for id in issue_ids.split(",")]

            result = await self.client.request(
                "POST",
                "/api/v1/security/auto-remediate",
                data={"issue_ids": issue_list, "auto_apply": True}
            )

            return self.format_response(result, "Security Remediation Results")

        @self.mcp.tool()
        async def performance_audit(ctx: Context) -> str:
            """
            Audit system performance and identify bottlenecks.

            Analyzes:
            - API endpoint latency
            - Database query performance
            - Memory usage patterns
            - CPU utilization
            - Network I/O

            Returns:
                Performance audit report with optimization recommendations
            """
            result = await self.client.request(
                "POST",
                "/api/v1/agents/performance/audit"
            )

            return self.format_response(result, "Performance Audit Results")

        @self.mcp.tool()
        async def database_optimize(ctx: Context) -> str:
            """
            Optimize database performance.

            Performs:
            - Index analysis and optimization
            - Query performance tuning
            - Connection pool optimization
            - Vacuum and analyze operations
            - Statistics update

            Returns:
                Database optimization results
            """
            result = await self.client.request(
                "POST",
                "/api/v1/agents/database/optimize"
            )

            return self.format_response(result, "Database Optimization Results")

        # Add health endpoint
        @self.mcp.tool()
        async def infrastructure_health(ctx: Context) -> str:
            """
            Get overall infrastructure health status.

            Returns:
                Complete health dashboard with all infrastructure metrics
            """
            result = await self.client.request(
                "GET",
                "/api/v1/monitoring/status"
            )

            output = ["# Infrastructure Health Dashboard\n"]
            output.append(f"**Server Uptime:** {self.get_uptime()}\n")

            if "error" not in result:
                output.append("## System Status")
                output.append(f"**Status:** {result.get('status', 'Unknown')}")
                output.append(f"**API Health:** {result.get('api_health', 'Unknown')}")
                output.append(f"\n## Metrics")
                metrics = result.get('metrics', {})
                for key, value in metrics.items():
                    output.append(f"- **{key}:** {value}")
            else:
                output.append(f"\n❌ Error: {result.get('error')}")

            return "\n".join(output)


# Initialize and run server
if __name__ == "__main__":
    server = InfrastructureMCPServer(config)
    server.run()
