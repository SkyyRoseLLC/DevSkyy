#!/usr/bin/env python3
"""
Base MCP Server Classes
Shared functionality for all DevSkyy MCP servers
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastmcp import FastMCP
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MCPServerConfig(BaseModel):
    """Configuration for MCP server"""
    name: str
    version: str = "1.0.0"
    description: str
    api_url: str
    api_key: str
    port: int
    timeout: float = 60.0
    max_retries: int = 3
    category: str


class DevSkyyAPIClient:
    """Base API client for all MCP servers"""

    def __init__(self, api_url: str, api_key: str, timeout: float = 60.0):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    async def request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make API request with error handling"""
        url = f"{self.api_url}{endpoint}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                logger.info(f"Making {method} request to {url}")
                response = await client.request(
                    method=method,
                    url=url,
                    headers=self.headers,
                    json=data,
                    params=params
                )
                response.raise_for_status()
                result = response.json()
                logger.info(f"Request successful: {method} {endpoint}")
                return result
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
                return {
                    "error": f"HTTP {e.response.status_code}",
                    "detail": e.response.text,
                    "endpoint": endpoint
                }
            except httpx.RequestError as e:
                logger.error(f"Request error: {str(e)}")
                return {
                    "error": "Request failed",
                    "detail": str(e),
                    "endpoint": endpoint
                }
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}")
                return {
                    "error": "Invalid JSON response",
                    "detail": str(e),
                    "endpoint": endpoint
                }
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                return {
                    "error": "Unexpected error",
                    "detail": str(e),
                    "endpoint": endpoint
                }


class BaseMCPServer:
    """Base class for all MCP servers"""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.mcp = FastMCP(config.name, version=config.version)
        self.client = DevSkyyAPIClient(
            api_url=config.api_url,
            api_key=config.api_key,
            timeout=config.timeout
        )
        self.start_time = datetime.now()
        logger.info(f"Initialized {config.name} MCP server")

    def format_response(self, result: Dict[str, Any], title: str) -> str:
        """Format API response as markdown"""
        output = [f"# {title}\n"]
        output.append(f"**Timestamp:** {datetime.now().isoformat()}\n")

        if "error" in result:
            output.append(f"\n## ❌ Error\n")
            output.append(f"**Error:** {result['error']}")
            if "detail" in result:
                output.append(f"\n**Details:** {result['detail']}")
            return "\n".join(output)

        # Format successful response
        for key, value in result.items():
            if isinstance(value, (list, dict)):
                output.append(f"\n## {key.replace('_', ' ').title()}")
                if isinstance(value, list) and value:
                    for item in value:
                        if isinstance(item, dict):
                            output.append(f"\n**Item:**")
                            for k, v in item.items():
                                output.append(f"- {k}: {v}")
                        else:
                            output.append(f"- {item}")
                elif isinstance(value, dict):
                    for k, v in value.items():
                        output.append(f"- **{k}:** {v}")
            else:
                output.append(f"\n**{key.replace('_', ' ').title()}:** {value}")

        return "\n".join(output)

    def get_uptime(self) -> str:
        """Get server uptime"""
        delta = datetime.now() - self.start_time
        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours}h {minutes}m {seconds}s"

    def print_banner(self):
        """Print startup banner"""
        print("=" * 70)
        print(f"   {self.config.name.upper()} - {self.config.description}")
        print("=" * 70)
        print()
        print("✅ Configuration:")
        print(f"   Server: {self.config.name}")
        print(f"   Version: {self.config.version}")
        print(f"   Category: {self.config.category}")
        print(f"   API URL: {self.config.api_url}")
        print(f"   API Key: {'Set ✓' if self.config.api_key else 'Not Set ✗'}")
        print(f"   Port: {self.config.port}")
        print()

    def run(self):
        """Run the MCP server"""
        self.print_banner()
        logger.info(f"Starting {self.config.name} MCP server...")
        self.mcp.run()


def load_env_config(category: str, port: int) -> MCPServerConfig:
    """Load MCP server configuration from environment"""
    api_url = os.getenv("DEVSKYY_API_URL", "http://localhost:8000")
    api_key = os.getenv("DEVSKYY_API_KEY", "")

    if not api_key:
        logger.warning("DEVSKYY_API_KEY not set. Some features may not work.")

    return MCPServerConfig(
        name=f"devskyy_{category}",
        description=f"DevSkyy {category.title()} MCP Server",
        api_url=api_url,
        api_key=api_key,
        port=port,
        category=category
    )
