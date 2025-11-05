#!/usr/bin/env python3
"""
MCP Service Discovery
Registry and health monitoring for all MCP servers
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class MCPServerInfo(BaseModel):
    """Information about an MCP server"""
    name: str
    category: str
    host: str
    port: int
    status: str = "unknown"  # unknown, healthy, unhealthy, down
    last_check: Optional[str] = None
    uptime: Optional[str] = None
    tools_count: int = 0
    error_message: Optional[str] = None


class ServiceRegistry:
    """Registry of all MCP servers"""

    def __init__(self, registry_file: str = "mcp_servers/config/registry.json"):
        self.registry_file = Path(registry_file)
        self.servers: Dict[str, MCPServerInfo] = {}
        self.load_registry()

    def load_registry(self):
        """Load registry from file"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                    for server_data in data.get("servers", []):
                        server = MCPServerInfo(**server_data)
                        self.servers[server.name] = server
                logger.info(f"Loaded {len(self.servers)} servers from registry")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")

    def save_registry(self):
        """Save registry to file"""
        try:
            self.registry_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "last_updated": datetime.now().isoformat(),
                "servers": [server.dict() for server in self.servers.values()]
            }
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.servers)} servers to registry")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    def register(self, server_info: MCPServerInfo):
        """Register a new MCP server"""
        self.servers[server_info.name] = server_info
        self.save_registry()
        logger.info(f"Registered server: {server_info.name}")

    def unregister(self, server_name: str):
        """Unregister an MCP server"""
        if server_name in self.servers:
            del self.servers[server_name]
            self.save_registry()
            logger.info(f"Unregistered server: {server_name}")

    def get_server(self, name: str) -> Optional[MCPServerInfo]:
        """Get server info by name"""
        return self.servers.get(name)

    def get_all_servers(self) -> List[MCPServerInfo]:
        """Get all registered servers"""
        return list(self.servers.values())

    def get_servers_by_category(self, category: str) -> List[MCPServerInfo]:
        """Get servers by category"""
        return [s for s in self.servers.values() if s.category == category]


class HealthChecker:
    """Health checker for MCP servers"""

    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self.check_interval = 30  # seconds

    async def check_server(self, server_info: MCPServerInfo) -> MCPServerInfo:
        """Check health of a single server"""
        try:
            url = f"http://{server_info.host}:{server_info.port}/health"
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    server_info.status = "healthy"
                    server_info.uptime = data.get("uptime", "unknown")
                    server_info.tools_count = data.get("tools_count", 0)
                    server_info.error_message = None
                else:
                    server_info.status = "unhealthy"
                    server_info.error_message = f"HTTP {response.status_code}"
        except httpx.ConnectError:
            server_info.status = "down"
            server_info.error_message = "Connection refused"
        except httpx.TimeoutException:
            server_info.status = "unhealthy"
            server_info.error_message = "Timeout"
        except Exception as e:
            server_info.status = "unhealthy"
            server_info.error_message = str(e)

        server_info.last_check = datetime.now().isoformat()
        return server_info

    async def check_all_servers(self):
        """Check health of all registered servers"""
        logger.info("Checking health of all servers...")
        tasks = [
            self.check_server(server)
            for server in self.registry.get_all_servers()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        healthy_count = sum(1 for s in results if isinstance(s, MCPServerInfo) and s.status == "healthy")
        total_count = len(results)

        logger.info(f"Health check complete: {healthy_count}/{total_count} servers healthy")
        self.registry.save_registry()

        return results

    async def start_monitoring(self):
        """Start continuous health monitoring"""
        logger.info(f"Starting health monitoring (interval: {self.check_interval}s)")
        while True:
            await self.check_all_servers()
            await asyncio.sleep(self.check_interval)


# Pre-configured server registry
DEFAULT_SERVERS = [
    MCPServerInfo(
        name="devskyy_infrastructure",
        category="infrastructure",
        host="localhost",
        port=50051
    ),
    MCPServerInfo(
        name="devskyy_ecommerce",
        category="ecommerce",
        host="localhost",
        port=50052
    ),
    MCPServerInfo(
        name="devskyy_wordpress",
        category="wordpress",
        host="localhost",
        port=50053
    ),
    MCPServerInfo(
        name="devskyy_ml",
        category="ml",
        host="localhost",
        port=50054
    ),
    MCPServerInfo(
        name="devskyy_marketing",
        category="marketing",
        host="localhost",
        port=50055
    ),
    MCPServerInfo(
        name="devskyy_orchestration",
        category="orchestration",
        host="localhost",
        port=50056
    ),
]


def initialize_registry():
    """Initialize registry with default servers"""
    registry = ServiceRegistry()
    for server_info in DEFAULT_SERVERS:
        registry.register(server_info)
    return registry


if __name__ == "__main__":
    # Initialize and run health checker
    logging.basicConfig(level=logging.INFO)
    registry = initialize_registry()
    checker = HealthChecker(registry)

    print("=" * 70)
    print("MCP Service Registry & Health Checker")
    print("=" * 70)
    print(f"\nRegistered Servers: {len(registry.get_all_servers())}")
    for server in registry.get_all_servers():
        print(f"  - {server.name} ({server.category}) - {server.host}:{server.port}")
    print()

    # Run health check
    asyncio.run(checker.check_all_servers())

    print("\nHealth Check Results:")
    for server in registry.get_all_servers():
        status_emoji = {
            "healthy": "✅",
            "unhealthy": "⚠️",
            "down": "❌",
            "unknown": "❓"
        }.get(server.status, "❓")
        print(f"  {status_emoji} {server.name}: {server.status}")
        if server.error_message:
            print(f"     Error: {server.error_message}")
