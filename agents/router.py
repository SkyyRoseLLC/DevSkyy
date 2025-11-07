#!/usr/bin/env python3
"""
Agent Task Routing System
Routes tasks to appropriate specialized agents based on capabilities

References: /Users/coreyfoster/Desktop/CLAUDE_20-10_MASTER.md
Truth Protocol Compliance: Rules 1, 4, 10
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import re

# Add parent directory to path for standalone execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.loader import AgentConfigLoader, AgentType, AgentConfiguration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Task type enumeration for routing"""
    # Code Quality & Security
    CODE_AUDIT = "code_audit"
    CODE_REFACTOR = "code_refactor"
    SECURITY_SCAN = "security_scan"
    TEST_GENERATION = "test_generation"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"

    # Growth & Marketing
    WORDPRESS_THEME = "wordpress_theme"
    LANDING_PAGE = "landing_page"
    AB_TESTING = "ab_testing"
    SEO_OPTIMIZATION = "seo_optimization"
    CONVERSION_OPTIMIZATION = "conversion_optimization"

    # Data & Analytics
    DATA_ANALYSIS = "data_analysis"
    KPI_DASHBOARD = "kpi_dashboard"
    PREDICTIVE_ANALYTICS = "predictive_analytics"
    PROMPT_ROUTING = "prompt_routing"
    EVALUATION_HARNESS = "evaluation_harness"

    # Visual Content
    IMAGE_GENERATION = "image_generation"
    IMAGE_UPSCALING = "image_upscaling"
    VIDEO_AUTOMATION = "video_automation"
    PRODUCT_PHOTOGRAPHY = "product_photography"
    BRAND_ASSETS = "brand_assets"


@dataclass
class TaskContext:
    """Context information for a task"""
    task_type: TaskType
    description: str
    priority: int = 1  # 1-5, 5 being highest
    requirements: Dict[str, Any] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.requirements is None:
            self.requirements = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RoutingDecision:
    """Result of routing decision"""
    agent_id: str
    agent_name: str
    agent_type: AgentType
    confidence: float  # 0.0 - 1.0
    reasoning: str
    orchestration_command: str
    estimated_slo: Dict[str, Any]


class AgentRouter:
    """
    Routes tasks to appropriate specialized agents

    Usage:
        router = AgentRouter()
        task = TaskContext(
            task_type=TaskType.CODE_AUDIT,
            description="Audit FastAPI security implementation",
            priority=5
        )
        decision = router.route_task(task)
        print(f"Route to: {decision.agent_name}")
    """

    def __init__(self, config_loader: Optional[AgentConfigLoader] = None):
        """
        Initialize agent router

        Args:
            config_loader: AgentConfigLoader instance (creates new if None)
        """
        self.loader = config_loader or AgentConfigLoader()
        self.agents = self.loader.load_all_agents()

        # Build routing tables
        self._task_to_agent_type = self._build_task_type_mapping()
        self._capability_keywords = self._build_capability_keywords()

        logger.info(f"AgentRouter initialized with {len(self.agents)} agents")

    def _build_task_type_mapping(self) -> Dict[TaskType, AgentType]:
        """
        Build mapping from task types to agent types

        Returns:
            Dict mapping TaskType to AgentType
        """
        return {
            # Code Quality & Security → Professors of Code
            TaskType.CODE_AUDIT: AgentType.CODE_QUALITY_SECURITY,
            TaskType.CODE_REFACTOR: AgentType.CODE_QUALITY_SECURITY,
            TaskType.SECURITY_SCAN: AgentType.CODE_QUALITY_SECURITY,
            TaskType.TEST_GENERATION: AgentType.CODE_QUALITY_SECURITY,
            TaskType.PERFORMANCE_OPTIMIZATION: AgentType.CODE_QUALITY_SECURITY,

            # Growth & Marketing → Growth Stack
            TaskType.WORDPRESS_THEME: AgentType.GROWTH_MARKETING_AUTOMATION,
            TaskType.LANDING_PAGE: AgentType.GROWTH_MARKETING_AUTOMATION,
            TaskType.AB_TESTING: AgentType.GROWTH_MARKETING_AUTOMATION,
            TaskType.SEO_OPTIMIZATION: AgentType.GROWTH_MARKETING_AUTOMATION,
            TaskType.CONVERSION_OPTIMIZATION: AgentType.GROWTH_MARKETING_AUTOMATION,

            # Data & Analytics → Data & Reasoning
            TaskType.DATA_ANALYSIS: AgentType.DATA_ANALYSIS_INTELLIGENCE,
            TaskType.KPI_DASHBOARD: AgentType.DATA_ANALYSIS_INTELLIGENCE,
            TaskType.PREDICTIVE_ANALYTICS: AgentType.DATA_ANALYSIS_INTELLIGENCE,
            TaskType.PROMPT_ROUTING: AgentType.DATA_ANALYSIS_INTELLIGENCE,
            TaskType.EVALUATION_HARNESS: AgentType.DATA_ANALYSIS_INTELLIGENCE,

            # Visual Content → Visual Foundry
            TaskType.IMAGE_GENERATION: AgentType.VISUAL_CONTENT_GENERATION,
            TaskType.IMAGE_UPSCALING: AgentType.VISUAL_CONTENT_GENERATION,
            TaskType.VIDEO_AUTOMATION: AgentType.VISUAL_CONTENT_GENERATION,
            TaskType.PRODUCT_PHOTOGRAPHY: AgentType.VISUAL_CONTENT_GENERATION,
            TaskType.BRAND_ASSETS: AgentType.VISUAL_CONTENT_GENERATION,
        }

    def _build_capability_keywords(self) -> Dict[AgentType, List[str]]:
        """
        Build keyword lists for each agent type for fuzzy matching

        Returns:
            Dict mapping AgentType to list of keywords
        """
        return {
            AgentType.CODE_QUALITY_SECURITY: [
                'code', 'audit', 'security', 'test', 'refactor', 'performance',
                'vulnerability', 'compliance', 'rbac', 'jwt', 'encryption',
                'pytest', 'mypy', 'bandit', 'fix', 'scan', 'review'
            ],
            AgentType.GROWTH_MARKETING_AUTOMATION: [
                'wordpress', 'theme', 'landing', 'page', 'marketing', 'campaign',
                'seo', 'conversion', 'analytics', 'ab', 'test', 'elementor',
                'woocommerce', 'social', 'media', 'email', 'funnel'
            ],
            AgentType.DATA_ANALYSIS_INTELLIGENCE: [
                'data', 'analysis', 'kpi', 'dashboard', 'metric', 'prediction',
                'forecast', 'prompt', 'routing', 'evaluation', 'statistical',
                'model', 'sql', 'database', 'report', 'insight'
            ],
            AgentType.VISUAL_CONTENT_GENERATION: [
                'image', 'video', 'visual', 'upscale', 'generate', 'photo',
                'brand', 'asset', 'design', 'graphic', 'creative', 'render',
                'stable', 'diffusion', 'dalle', 'midjourney'
            ]
        }

    def route_task(self, task: TaskContext) -> RoutingDecision:
        """
        Route a task to the appropriate agent

        Args:
            task: TaskContext with task information

        Returns:
            RoutingDecision with routing information
        """
        # Step 1: Direct mapping from task type
        if task.task_type in self._task_to_agent_type:
            agent_type = self._task_to_agent_type[task.task_type]
            confidence = 1.0
            reasoning = f"Direct task type mapping: {task.task_type.value} → {agent_type.value}"
        else:
            # Step 2: Fuzzy matching based on description keywords
            agent_type, confidence = self._fuzzy_match_agent_type(task.description)
            reasoning = f"Fuzzy keyword matching on description (confidence: {confidence:.2f})"

        # Step 3: Find agent of the determined type
        agent_id = None
        for aid, agent_config in self.agents.items():
            if agent_config.agent_type == agent_type:
                agent_id = aid
                break

        if agent_id is None:
            # Fallback: use first available agent
            agent_id = list(self.agents.keys())[0]
            agent_config = self.agents[agent_id]
            reasoning = f"FALLBACK: No agent found for type {agent_type.value}, using {agent_id}"
            confidence = 0.5
            logger.warning(f"No agent found for type {agent_type.value}, falling back to {agent_id}")
        else:
            agent_config = self.agents[agent_id]

        # Step 4: Select appropriate orchestration command
        command = self._select_orchestration_command(task, agent_config)

        # Step 5: Estimate SLOs
        estimated_slo = self._estimate_slo(task, agent_config)

        decision = RoutingDecision(
            agent_id=agent_id,
            agent_name=agent_config.agent_name,
            agent_type=agent_config.agent_type,
            confidence=confidence,
            reasoning=reasoning,
            orchestration_command=command,
            estimated_slo=estimated_slo
        )

        logger.info(
            f"Routed task '{task.task_type.value}' to {decision.agent_name} "
            f"(confidence: {decision.confidence:.2f})"
        )

        return decision

    def _fuzzy_match_agent_type(self, description: str) -> Tuple[AgentType, float]:
        """
        Fuzzy match agent type based on description keywords

        Args:
            description: Task description text

        Returns:
            Tuple of (AgentType, confidence_score)
        """
        description_lower = description.lower()
        scores = {}

        for agent_type, keywords in self._capability_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in description_lower:
                    score += 1

            # Normalize score
            scores[agent_type] = score / len(keywords)

        if not scores:
            # Default to code quality if no matches
            return AgentType.CODE_QUALITY_SECURITY, 0.3

        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]

        # If confidence too low, use code quality as safe default
        if confidence < 0.1:
            return AgentType.CODE_QUALITY_SECURITY, 0.3

        return best_type, confidence

    def _select_orchestration_command(
        self,
        task: TaskContext,
        agent: AgentConfiguration
    ) -> str:
        """
        Select appropriate orchestration command for task

        Args:
            task: Task context
            agent: Agent configuration

        Returns:
            Orchestration command name
        """
        # Priority-based command selection
        if task.priority >= 4:
            # High priority: prefer PLAN for comprehensive approach
            if 'PLAN' in agent.orchestration_commands:
                return 'PLAN'

        # Task type specific command preferences
        if task.task_type in [TaskType.CODE_AUDIT, TaskType.SECURITY_SCAN]:
            if 'REVIEW' in agent.orchestration_commands:
                return 'REVIEW'
            if 'TEST' in agent.orchestration_commands:
                return 'TEST'

        if task.task_type in [TaskType.DATA_ANALYSIS, TaskType.KPI_DASHBOARD]:
            if 'MONITOR' in agent.orchestration_commands:
                return 'MONITOR'

        # Default: BUILD for implementation tasks
        if 'BUILD' in agent.orchestration_commands:
            return 'BUILD'

        # Fallback: first available command
        if agent.orchestration_commands:
            return list(agent.orchestration_commands.keys())[0]

        return 'EXECUTE'

    def _estimate_slo(
        self,
        task: TaskContext,
        agent: AgentConfiguration
    ) -> Dict[str, Any]:
        """
        Estimate SLO compliance for task

        Args:
            task: Task context
            agent: Agent configuration

        Returns:
            Dict with estimated SLO values
        """
        slo = agent.performance_slos
        estimated = {}

        # Priority adjustments
        priority_multiplier = 1.0 + (task.priority - 1) * 0.2

        if slo.p95_latency_ms:
            estimated['p95_latency_ms'] = int(slo.p95_latency_ms * priority_multiplier)

        if slo.test_coverage_percent:
            estimated['test_coverage_percent'] = slo.test_coverage_percent

        if slo.error_rate_percent:
            estimated['error_rate_percent'] = slo.error_rate_percent

        estimated['priority'] = task.priority
        estimated['meets_slo'] = True  # Optimistic default

        return estimated

    def route_multiple_tasks(
        self,
        tasks: List[TaskContext]
    ) -> List[RoutingDecision]:
        """
        Route multiple tasks to appropriate agents

        Args:
            tasks: List of TaskContext objects

        Returns:
            List of RoutingDecision objects
        """
        decisions = []
        for task in tasks:
            try:
                decision = self.route_task(task)
                decisions.append(decision)
            except Exception as e:
                logger.error(f"Failed to route task {task.task_type.value}: {e}")
                continue

        return decisions

    def get_agent_workload(self) -> Dict[str, int]:
        """
        Get workload distribution across agents

        Returns:
            Dict mapping agent_id to task count
        """
        # This would track actual task assignments in production
        # For now, return empty dict
        return {agent_id: 0 for agent_id in self.agents.keys()}

    def suggest_agent_for_natural_language(self, request: str) -> RoutingDecision:
        """
        Route based on natural language request

        Args:
            request: Natural language task description

        Returns:
            RoutingDecision
        """
        # Analyze request and infer task type
        request_lower = request.lower()

        # Pattern matching for task type inference
        if any(kw in request_lower for kw in ['audit', 'security', 'test', 'fix', 'refactor']):
            task_type = TaskType.CODE_AUDIT
        elif any(kw in request_lower for kw in ['wordpress', 'theme', 'landing', 'page']):
            task_type = TaskType.WORDPRESS_THEME
        elif any(kw in request_lower for kw in ['data', 'analysis', 'dashboard', 'kpi']):
            task_type = TaskType.DATA_ANALYSIS
        elif any(kw in request_lower for kw in ['image', 'video', 'visual', 'generate']):
            task_type = TaskType.IMAGE_GENERATION
        else:
            # Default to code audit for uncertain requests
            task_type = TaskType.CODE_AUDIT

        task = TaskContext(
            task_type=task_type,
            description=request,
            priority=3  # Medium priority default
        )

        return self.route_task(task)


# Example usage
if __name__ == "__main__":
    # Initialize router
    router = AgentRouter()

    print("\n" + "=" * 70)
    print("Agent Task Routing Examples")
    print("=" * 70)

    # Example 1: Code audit task
    print("\n### Example 1: Code Audit ###")
    task1 = TaskContext(
        task_type=TaskType.CODE_AUDIT,
        description="Audit FastAPI security implementation for JWT vulnerabilities",
        priority=5,
        requirements={'coverage': 90, 'tools': ['bandit', 'safety']}
    )
    decision1 = router.route_task(task1)
    print(f"Task Type: {task1.task_type.value}")
    print(f"Routed To: {decision1.agent_name} ({decision1.agent_id})")
    print(f"Confidence: {decision1.confidence:.2f}")
    print(f"Reasoning: {decision1.reasoning}")
    print(f"Command: {decision1.orchestration_command}")
    print(f"Estimated SLO: {decision1.estimated_slo}")

    # Example 2: WordPress theme
    print("\n### Example 2: WordPress Theme ###")
    task2 = TaskContext(
        task_type=TaskType.WORDPRESS_THEME,
        description="Create luxury fashion WordPress theme with Elementor",
        priority=3
    )
    decision2 = router.route_task(task2)
    print(f"Task Type: {task2.task_type.value}")
    print(f"Routed To: {decision2.agent_name}")
    print(f"Command: {decision2.orchestration_command}")

    # Example 3: Data analysis
    print("\n### Example 3: Data Analysis ###")
    task3 = TaskContext(
        task_type=TaskType.KPI_DASHBOARD,
        description="Build revenue and conversion KPI dashboard",
        priority=4
    )
    decision3 = router.route_task(task3)
    print(f"Task Type: {task3.task_type.value}")
    print(f"Routed To: {decision3.agent_name}")
    print(f"Command: {decision3.orchestration_command}")

    # Example 4: Image generation
    print("\n### Example 4: Image Generation ###")
    task4 = TaskContext(
        task_type=TaskType.IMAGE_GENERATION,
        description="Generate high-end product photography for luxury handbags",
        priority=4
    )
    decision4 = router.route_task(task4)
    print(f"Task Type: {task4.task_type.value}")
    print(f"Routed To: {decision4.agent_name}")
    print(f"Command: {decision4.orchestration_command}")

    # Example 5: Natural language routing
    print("\n### Example 5: Natural Language Routing ###")
    nl_requests = [
        "I need help auditing my Python code for security issues",
        "Create a landing page for our new product launch",
        "Analyze our customer data and create a dashboard",
        "Generate product images for our e-commerce site"
    ]

    for request in nl_requests:
        decision = router.suggest_agent_for_natural_language(request)
        print(f"\nRequest: '{request}'")
        print(f"→ {decision.agent_name} (confidence: {decision.confidence:.2f})")

    # Example 6: Multiple tasks
    print("\n### Example 6: Batch Routing ###")
    tasks = [
        TaskContext(TaskType.CODE_AUDIT, "Security audit", priority=5),
        TaskContext(TaskType.LANDING_PAGE, "Create landing page", priority=3),
        TaskContext(TaskType.DATA_ANALYSIS, "Revenue analysis", priority=4),
    ]

    decisions = router.route_multiple_tasks(tasks)
    print(f"\nRouted {len(decisions)} tasks:")
    for i, d in enumerate(decisions, 1):
        print(f"  {i}. {d.agent_name} - {d.orchestration_command}")

    # Show agent workload
    print("\n### Agent Workload ###")
    workload = router.get_agent_workload()
    for agent_id, count in workload.items():
        print(f"  {agent_id}: {count} tasks")
