#!/usr/bin/env python3
"""
Agent Orchestration Integration Examples
Demonstrates programmatic agent loading and task routing

References: /Users/coreyfoster/Desktop/CLAUDE_20-10_MASTER.md
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import (
    AgentConfigLoader,
    AgentRouter,
    TaskType,
    TaskContext,
    AgentType
)


def example_1_load_and_inspect_agents():
    """Example 1: Load and inspect agent configurations"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Load and Inspect Agents")
    print("=" * 70 + "\n")

    # Initialize loader
    loader = AgentConfigLoader()

    # Load all agents
    agents = loader.load_all_agents()
    print(f"📦 Loaded {len(agents)} agents\n")

    # Inspect each agent
    for agent_id, agent in agents.items():
        print(f"🤖 {agent.agent_name}")
        print(f"   ID: {agent_id}")
        print(f"   Type: {agent.agent_type.value}")
        print(f"   Status: {agent.status.value}")
        print(f"   Composition: {agent.composition['primary_ai']} + {agent.composition['secondary_ai']}")
        print(f"   Commands: {', '.join(agent.orchestration_commands.keys())}")
        print(f"   SLO P95: {agent.performance_slos.p95_latency_ms}ms")
        print()


def example_2_validate_truth_protocol_compliance():
    """Example 2: Validate agents against Truth Protocol"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Validate Truth Protocol Compliance")
    print("=" * 70 + "\n")

    loader = AgentConfigLoader()

    for agent_id in loader.get_agent_ids():
        validation = loader.validate_agent(agent_id)

        status = "✅ PASS" if validation['valid'] else "❌ FAIL"
        print(f"{status} {agent_id}")

        if validation['errors']:
            for error in validation['errors']:
                print(f"    ERROR: {error}")

        if validation['warnings']:
            for warning in validation['warnings']:
                print(f"    WARNING: {warning}")

        print()


def example_3_route_specific_tasks():
    """Example 3: Route specific tasks to agents"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Route Specific Tasks")
    print("=" * 70 + "\n")

    router = AgentRouter()

    tasks = [
        TaskContext(
            task_type=TaskType.CODE_AUDIT,
            description="Audit FastAPI endpoints for security vulnerabilities",
            priority=5,
            requirements={'coverage': 90, 'tools': ['bandit', 'safety']}
        ),
        TaskContext(
            task_type=TaskType.WORDPRESS_THEME,
            description="Create luxury fashion WordPress theme with Elementor Pro",
            priority=4,
            requirements={'framework': 'Elementor', 'responsive': True}
        ),
        TaskContext(
            task_type=TaskType.DATA_ANALYSIS,
            description="Analyze customer conversion funnel and create dashboard",
            priority=3,
            requirements={'statistical_significance': 0.95}
        ),
        TaskContext(
            task_type=TaskType.IMAGE_UPSCALING,
            description="Upscale product images to 4K resolution",
            priority=4,
            requirements={'scale_factor': 4, 'quality': 'high'}
        )
    ]

    print("🎯 Routing Tasks:\n")

    for i, task in enumerate(tasks, 1):
        decision = router.route_task(task)

        print(f"{i}. Task: {task.task_type.value}")
        print(f"   Description: {task.description}")
        print(f"   Priority: {'⭐' * task.priority}")
        print(f"   → Routed to: {decision.agent_name}")
        print(f"   → Command: {decision.orchestration_command}")
        print(f"   → Confidence: {decision.confidence:.0%}")
        print(f"   → Est. Latency: {decision.estimated_slo.get('p95_latency_ms', 'N/A')}ms")
        print()


def example_4_natural_language_routing():
    """Example 4: Route natural language requests"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Natural Language Routing")
    print("=" * 70 + "\n")

    router = AgentRouter()

    requests = [
        "I need to audit my Python codebase for security vulnerabilities and generate comprehensive tests",
        "Create a beautiful landing page for our new luxury handbag collection with A/B testing",
        "Analyze our sales data from the last quarter and create a KPI dashboard for executives",
        "Generate high-quality product images for our e-commerce site and upscale them to 4K",
        "Build a WordPress theme for our fashion blog with SEO optimization",
        "Set up email marketing campaign with customer segmentation and analytics"
    ]

    print("💬 Natural Language Routing:\n")

    for i, request in enumerate(requests, 1):
        decision = router.suggest_agent_for_natural_language(request)

        print(f"{i}. Request:")
        print(f"   \"{request}\"")
        print(f"   → Agent: {decision.agent_name}")
        print(f"   → Confidence: {decision.confidence:.0%}")
        print(f"   → Reasoning: {decision.reasoning}")
        print()


def example_5_agent_capabilities_query():
    """Example 5: Query agent capabilities"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Query Agent Capabilities")
    print("=" * 70 + "\n")

    loader = AgentConfigLoader()

    # Find agents by type
    print("📋 Agents by Type:\n")

    for agent_type in AgentType:
        agent_ids = loader.get_agent_by_type(agent_type)
        print(f"{agent_type.value}:")
        if agent_ids:
            for agent_id in agent_ids:
                agent = loader.load_agent(agent_id)
                primary_functions = agent.capabilities.get('primary_functions', [])
                print(f"  • {agent.agent_name}")
                for func in primary_functions[:3]:  # Show first 3 functions
                    print(f"    - {func}")
        else:
            print("  (No agents)")
        print()


def example_6_orchestration_workflow():
    """Example 6: Complete orchestration workflow simulation"""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Complete Orchestration Workflow")
    print("=" * 70 + "\n")

    loader = AgentConfigLoader()
    router = AgentRouter()

    # User request
    request = "I need to build a complete e-commerce platform with security audit"

    print(f"👤 User Request:")
    print(f"   \"{request}\"\n")

    # Break down into subtasks
    subtasks = [
        TaskContext(
            task_type=TaskType.CODE_AUDIT,
            description="Security audit of e-commerce backend",
            priority=5
        ),
        TaskContext(
            task_type=TaskType.WORDPRESS_THEME,
            description="Create e-commerce theme with WooCommerce",
            priority=4
        ),
        TaskContext(
            task_type=TaskType.PRODUCT_PHOTOGRAPHY,
            description="Generate product images",
            priority=3
        ),
        TaskContext(
            task_type=TaskType.DATA_ANALYSIS,
            description="Set up analytics and KPI tracking",
            priority=3
        )
    ]

    print("📋 Workflow Breakdown:\n")

    # Route each subtask
    for i, task in enumerate(subtasks, 1):
        decision = router.route_task(task)
        agent = loader.load_agent(decision.agent_id)

        print(f"Step {i}: {task.description}")
        print(f"  Agent: {decision.agent_name}")
        print(f"  Command: {decision.orchestration_command}")

        # Get command details
        if decision.orchestration_command in agent.orchestration_commands:
            cmd = agent.orchestration_commands[decision.orchestration_command]
            print(f"  Mode: {cmd.execution_mode}")
            if cmd.tools:
                print(f"  Tools: {', '.join(cmd.tools)}")

        print()

    # Show deliverables
    print("📦 Expected Deliverables:\n")

    unique_agents = set(router.route_task(t).agent_id for t in subtasks)
    for agent_id in unique_agents:
        agent = loader.load_agent(agent_id)
        print(f"{agent.agent_name}:")
        for deliverable in agent.deliverables_per_cycle[:3]:
            print(f"  • {deliverable}")
        print()


def example_7_slo_compliance_check():
    """Example 7: Check SLO compliance for tasks"""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: SLO Compliance Check")
    print("=" * 70 + "\n")

    loader = AgentConfigLoader()
    router = AgentRouter()

    task = TaskContext(
        task_type=TaskType.CODE_AUDIT,
        description="Critical security audit",
        priority=5
    )

    decision = router.route_task(task)
    agent = loader.load_agent(decision.agent_id)

    print(f"📊 SLO Compliance for: {agent.agent_name}\n")

    slo = agent.performance_slos

    print("Performance Targets:")
    if slo.p95_latency_ms:
        status = "✅" if slo.p95_latency_ms <= 200 else "⚠️"
        print(f"  {status} P95 Latency: {slo.p95_latency_ms}ms (target: ≤200ms)")

    if slo.test_coverage_percent:
        status = "✅" if slo.test_coverage_percent >= 90 else "⚠️"
        print(f"  {status} Test Coverage: {slo.test_coverage_percent}% (target: ≥90%)")

    if slo.error_rate_percent:
        status = "✅" if slo.error_rate_percent <= 0.5 else "⚠️"
        print(f"  {status} Error Rate: {slo.error_rate_percent}% (target: ≤0.5%)")

    print(f"  ✅ Secrets in Repo: {slo.secrets_in_repo} (target: 0)")

    print()


def example_8_quick_reference_usage():
    """Example 8: Use quick reference for common tasks"""
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Quick Reference Usage")
    print("=" * 70 + "\n")

    loader = AgentConfigLoader()
    quick_ref = loader.get_quick_reference()

    print("🔗 Quick Reference Mappings:\n")

    use_cases = {
        "code_quality": "Need code review and security audit",
        "marketing_growth": "Want to launch marketing campaign",
        "data_analytics": "Need data analysis and reporting",
        "visual_content": "Need images and videos generated"
    }

    for key, description in use_cases.items():
        if key in quick_ref:
            agent_id = quick_ref[key]
            agent = loader.load_agent(agent_id)
            print(f"{description}")
            print(f"  → Use: {agent.agent_name} ({agent_id})")
            print()


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("🤖 DevSkyy Agent Orchestration System Examples")
    print("   Truth Protocol 2.0 Compliance")
    print("=" * 70)

    try:
        example_1_load_and_inspect_agents()
        example_2_validate_truth_protocol_compliance()
        example_3_route_specific_tasks()
        example_4_natural_language_routing()
        example_5_agent_capabilities_query()
        example_6_orchestration_workflow()
        example_7_slo_compliance_check()
        example_8_quick_reference_usage()

        print("\n" + "=" * 70)
        print("✅ All examples completed successfully!")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
