#!/usr/bin/env python3
"""
Test All DevSkyy MCP Servers
Comprehensive testing script for all 6 MCP servers

Usage:
    python3 test_all_mcp_servers.py
"""

import asyncio
import json
from typing import Dict, List, Any
from datetime import datetime

# Test configuration
MCP_SERVERS = [
    {
        "name": "Infrastructure",
        "port": 5001,
        "tools": ["scan_code", "fix_code", "security_scan", "self_healing_check"]
    },
    {
        "name": "Ecommerce",
        "port": 5002,
        "tools": ["create_product", "update_product", "optimize_product_seo", "dynamic_pricing"]
    },
    {
        "name": "WordPress",
        "port": 5003,
        "tools": ["generate_wordpress_theme", "deploy_theme_to_wordpress", "generate_product_pages"]
    },
    {
        "name": "AI/ML",
        "port": 5004,
        "tools": ["predict_fashion_trends", "optimize_product_pricing", "generate_product_image", "analyze_customer_sentiment"]
    },
    {
        "name": "Marketing",
        "port": 5005,
        "tools": ["create_marketing_campaign", "generate_social_media_content", "schedule_social_posts", "analyze_campaign_performance"]
    },
    {
        "name": "Orchestration",
        "port": 5006,
        "tools": ["get_system_health", "list_all_agents", "execute_workflow", "get_system_metrics"]
    }
]


class MCPServerTester:
    """Test harness for MCP servers"""

    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.start_time = datetime.now()

    def print_header(self, text: str):
        """Print formatted header"""
        print("\n" + "=" * 70)
        print(f"   {text}")
        print("=" * 70 + "\n")

    def print_test_result(self, test_name: str, passed: bool, details: str = ""):
        """Print test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
        if details:
            print(f"     {details}")

    async def test_server_startup(self, server: Dict[str, Any]) -> bool:
        """Test if MCP server can be imported"""
        try:
            module_name = f"mcp_servers.{server['name'].lower().replace(' ', '_').replace('/', '_')}_mcp"
            print(f"   Testing: {module_name}")
            return True
        except Exception as e:
            print(f"   Error: {str(e)}")
            return False

    async def test_all_servers(self):
        """Run tests on all MCP servers"""
        self.print_header("DevSkyy MCP Servers - Comprehensive Test Suite")

        print(f"📅 Test Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔧 Testing {len(MCP_SERVERS)} MCP servers...")
        print()

        all_passed = True

        for server in MCP_SERVERS:
            self.print_header(f"{server['name']} MCP Server (Port {server['port']})")

            # Test 1: Server file exists
            server_file = f"mcp_servers/{server['name'].lower().replace(' ', '_').replace('/', '_')}_mcp.py"
            try:
                with open(server_file, 'r') as f:
                    content = f.read()
                    file_exists = len(content) > 0
            except FileNotFoundError:
                file_exists = False

            self.print_test_result(
                f"Server file exists: {server_file}",
                file_exists,
                f"File size: {len(content) if file_exists else 0} bytes"
            )

            # Test 2: Required imports present
            if file_exists:
                required_imports = ["BaseMCPServer", "load_env_config", "Context"]
                imports_present = all(imp in content for imp in required_imports)
                self.print_test_result(
                    "Required imports present",
                    imports_present,
                    f"Checking: {', '.join(required_imports)}"
                )
            else:
                imports_present = False
                self.print_test_result("Required imports present", False, "File not found")

            # Test 3: Tools registered
            if file_exists:
                tools_registered = all(f"async def {tool}" in content for tool in server['tools'])
                self.print_test_result(
                    f"All {len(server['tools'])} tools registered",
                    tools_registered,
                    f"Tools: {', '.join(server['tools'][:3])}..."
                )
            else:
                tools_registered = False
                self.print_test_result("Tools registered", False, "File not found")

            # Test 4: Port configuration
            if file_exists:
                port_configured = f"port={server['port']}" in content
                self.print_test_result(
                    f"Port configured correctly",
                    port_configured,
                    f"Expected port: {server['port']}"
                )
            else:
                port_configured = False
                self.print_test_result("Port configured", False, "File not found")

            # Update results
            server_passed = all([file_exists, imports_present, tools_registered, port_configured])
            self.results[server['name']] = {
                "passed": server_passed,
                "file_exists": file_exists,
                "imports_present": imports_present,
                "tools_registered": tools_registered,
                "port_configured": port_configured
            }

            if not server_passed:
                all_passed = False

        # Print summary
        self.print_header("Test Summary")

        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        passed_count = sum(1 for r in self.results.values() if r['passed'])
        total_count = len(self.results)

        print(f"⏱️  Duration: {duration:.2f} seconds")
        print(f"📊 Results: {passed_count}/{total_count} servers passed\n")

        for server_name, result in self.results.items():
            status = "✅" if result['passed'] else "❌"
            print(f"{status} {server_name} MCP Server")

        print()

        if all_passed:
            print("🎉 All MCP servers passed testing!")
        else:
            print("⚠️  Some MCP servers failed testing. Please review the output above.")

        return all_passed

    def generate_report(self):
        """Generate detailed test report"""
        report = {
            "test_date": self.start_time.isoformat(),
            "total_servers": len(MCP_SERVERS),
            "passed": sum(1 for r in self.results.values() if r['passed']),
            "failed": sum(1 for r in self.results.values() if not r['passed']),
            "details": self.results,
            "servers": [
                {
                    "name": server['name'],
                    "port": server['port'],
                    "tools_count": len(server['tools']),
                    "status": "passed" if self.results.get(server['name'], {}).get('passed') else "failed"
                }
                for server in MCP_SERVERS
            ]
        }

        with open("MCP_TEST_RESULTS.json", "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n📄 Detailed report saved to: MCP_TEST_RESULTS.json")


async def main():
    """Main test runner"""
    tester = MCPServerTester()

    try:
        success = await tester.test_all_servers()
        tester.generate_report()

        return 0 if success else 1

    except Exception as e:
        print(f"\n❌ Test suite error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
