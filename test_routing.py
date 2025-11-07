#!/usr/bin/env python3
"""Test enterprise agent routing system"""
from agents import AgentRouter, AgentConfigLoader, TaskRequest, TaskType

# Test batch routing for MCP efficiency
loader = AgentConfigLoader("config/agents")
router = AgentRouter(config_loader=loader)
tasks = [
    TaskRequest(TaskType.SECURITY_SCAN, "Scan for vulnerabilities", priority=90),
    TaskRequest(TaskType.CODE_GENERATION, "Fix syntax errors", priority=85),
    TaskRequest(TaskType.ML_TRAINING, "Train error model", priority=70)
]

results = router.route_multiple_tasks(tasks)
print(f"✓ Routed {len(results)} tasks successfully")
for i, result in enumerate(results):
    print(f"  Task {i+1}: {result.agent_name} (confidence: {result.confidence:.2f})")
    print(f"    Reasoning: {result.reasoning[:80]}...")
