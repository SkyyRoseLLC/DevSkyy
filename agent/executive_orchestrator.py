#!/usr/bin/env python3
"""
Executive Orchestrator (GOD MODE)
Top-level decision maker coordinating all agents, clusters, and infrastructure

Architecture Hierarchy:
    Executive Layer (THIS FILE)
        ├── Creative Cluster (Design, Marketing)
        ├── Commerce Cluster (E-commerce, Copy/Voice)
        ├── Brand Core AI (Multi-model intelligence)
        └── Data/Infrastructure Layers

References: /Users/coreyfoster/Desktop/CLAUDE_20-10_MASTER.md
Truth Protocol Compliance: All 15 rules
Version: 2.0.0
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import json

# DevSkyy imports
try:
    from agents.loader import AgentConfigLoader, AgentType
    from agents.router import AgentRouter, TaskType, TaskContext, RoutingDecision
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from agents.loader import AgentConfigLoader, AgentType
    from agents.router import AgentRouter, TaskType, TaskContext, RoutingDecision

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExecutiveDecision(Enum):
    """Executive-level decision types"""
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATE = "escalate"
    DEFER = "defer"
    DELEGATE = "delegate"


class ClusterType(Enum):
    """Cluster types in the architecture"""
    CREATIVE = "creative"
    COMMERCE = "commerce"
    BRAND_AI = "brand_ai"
    DATA_LAYER = "data_layer"
    INFRASTRUCTURE = "infrastructure"


@dataclass
class ExecutiveTask:
    """High-level task for executive decision making"""
    task_id: str
    description: str
    priority: int  # 1-5, 5 being critical
    cluster: ClusterType
    subtasks: List[TaskContext] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    estimated_duration_minutes: int = 60
    requires_approval: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutiveReport:
    """Executive summary report"""
    timestamp: datetime
    tasks_executed: int
    tasks_successful: int
    tasks_failed: int
    clusters_activated: List[ClusterType]
    resource_utilization: Dict[str, float]
    performance_metrics: Dict[str, Any]
    recommendations: List[str]


class ExecutiveOrchestrator:
    """
    GOD MODE - Executive Layer Orchestrator

    Responsibilities:
    1. High-level task decomposition
    2. Resource allocation across clusters
    3. Conflict resolution between agents
    4. Performance monitoring and optimization
    5. Strategic decision making
    6. System-wide coordination

    Usage:
        executive = ExecutiveOrchestrator()
        await executive.initialize()
        result = await executive.execute_mission(mission_description)
    """

    def __init__(self):
        """Initialize executive orchestrator"""
        self.agent_loader = AgentConfigLoader()
        self.agent_router = AgentRouter(self.agent_loader)

        # Cluster mappings
        self.cluster_mapping = self._build_cluster_mapping()

        # State tracking
        self.active_tasks: Dict[str, ExecutiveTask] = {}
        self.task_history: List[ExecutiveTask] = []
        self.performance_metrics: Dict[str, Any] = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'avg_execution_time_seconds': 0
        }

        # Resource limits (prevent overload)
        self.max_concurrent_tasks = 10
        self.max_tasks_per_cluster = 5

        logger.info("Executive Orchestrator initialized (GOD MODE)")

    def _build_cluster_mapping(self) -> Dict[ClusterType, List[AgentType]]:
        """
        Build mapping from clusters to agent types

        Returns:
            Dict mapping cluster types to lists of agent types
        """
        return {
            ClusterType.CREATIVE: [
                AgentType.VISUAL_CONTENT_GENERATION,  # Visual Foundry
                AgentType.GROWTH_MARKETING_AUTOMATION  # Marketing agents
            ],
            ClusterType.COMMERCE: [
                AgentType.GROWTH_MARKETING_AUTOMATION  # E-commerce (WordPress, WooCommerce)
            ],
            ClusterType.BRAND_AI: [
                AgentType.CODE_QUALITY_SECURITY,  # Professors of Code
                AgentType.DATA_ANALYSIS_INTELLIGENCE  # Data & Reasoning
            ],
            ClusterType.DATA_LAYER: [
                AgentType.DATA_ANALYSIS_INTELLIGENCE  # Vector DB, Knowledge Graph
            ],
            ClusterType.INFRASTRUCTURE: [
                AgentType.CODE_QUALITY_SECURITY  # Infrastructure monitoring
            ]
        }

    async def initialize(self):
        """Initialize executive orchestrator with system health check"""
        logger.info("Initializing Executive Orchestrator...")

        # Load all agent configurations
        agents = self.agent_loader.load_all_agents()
        logger.info(f"Loaded {len(agents)} agent configurations")

        # Validate Truth Protocol compliance
        for agent_id in self.agent_loader.get_agent_ids():
            validation = self.agent_loader.validate_agent(agent_id)
            if not validation['valid']:
                logger.warning(f"Agent {agent_id} failed validation: {validation['errors']}")

        logger.info("Executive Orchestrator initialized successfully")

    async def execute_mission(
        self,
        mission_description: str,
        priority: int = 3,
        requires_approval: bool = False
    ) -> ExecutiveReport:
        """
        Execute a high-level mission by decomposing into tasks and coordinating execution

        Args:
            mission_description: Natural language description of the mission
            priority: Mission priority (1-5, 5 being critical)
            requires_approval: Whether mission requires executive approval

        Returns:
            ExecutiveReport with execution summary
        """
        start_time = datetime.now()
        logger.info(f"Executing mission (priority {priority}): {mission_description}")

        # Step 1: Decompose mission into executive tasks
        executive_tasks = self._decompose_mission(mission_description, priority)

        # Step 2: Make executive decision (approve/reject)
        decision = self._make_decision(executive_tasks, requires_approval)

        if decision != ExecutiveDecision.APPROVED:
            logger.warning(f"Mission decision: {decision.value}")
            return self._create_report(start_time, executed=0, successful=0)

        # Step 3: Route tasks to appropriate clusters
        routing_plan = await self._create_routing_plan(executive_tasks)

        # Step 4: Execute tasks with resource management
        results = await self._execute_with_orchestration(routing_plan)

        # Step 5: Generate executive report
        report = self._create_report(
            start_time,
            executed=len(results),
            successful=sum(1 for r in results if r['success'])
        )

        logger.info(f"Mission complete: {report.tasks_successful}/{report.tasks_executed} successful")
        return report

    def _decompose_mission(
        self,
        mission_description: str,
        priority: int
    ) -> List[ExecutiveTask]:
        """
        Decompose high-level mission into executive tasks

        Args:
            mission_description: Mission description
            priority: Mission priority

        Returns:
            List of ExecutiveTask objects
        """
        # Use NLP to analyze mission and extract task components
        # For now, use keyword-based decomposition

        tasks = []
        mission_lower = mission_description.lower()

        # Creative cluster triggers
        if any(kw in mission_lower for kw in ['design', 'image', 'video', 'visual', 'brand', 'theme']):
            tasks.append(ExecutiveTask(
                task_id=f"creative_{len(tasks)}",
                description=f"Creative design for: {mission_description}",
                priority=priority,
                cluster=ClusterType.CREATIVE
            ))

        # Commerce cluster triggers
        if any(kw in mission_lower for kw in ['product', 'shop', 'ecommerce', 'inventory', 'order', 'payment']):
            tasks.append(ExecutiveTask(
                task_id=f"commerce_{len(tasks)}",
                description=f"Commerce operations for: {mission_description}",
                priority=priority,
                cluster=ClusterType.COMMERCE
            ))

        # Brand AI triggers
        if any(kw in mission_lower for kw in ['content', 'copy', 'voice', 'brand', 'marketing']):
            tasks.append(ExecutiveTask(
                task_id=f"brand_ai_{len(tasks)}",
                description=f"Brand AI generation for: {mission_description}",
                priority=priority,
                cluster=ClusterType.BRAND_AI
            ))

        # Data layer triggers
        if any(kw in mission_lower for kw in ['data', 'analysis', 'report', 'metrics', 'dashboard']):
            tasks.append(ExecutiveTask(
                task_id=f"data_{len(tasks)}",
                description=f"Data analysis for: {mission_description}",
                priority=priority,
                cluster=ClusterType.DATA_LAYER
            ))

        # If no specific cluster identified, default to Brand AI
        if not tasks:
            tasks.append(ExecutiveTask(
                task_id=f"default_{len(tasks)}",
                description=mission_description,
                priority=priority,
                cluster=ClusterType.BRAND_AI
            ))

        logger.info(f"Decomposed mission into {len(tasks)} executive tasks")
        return tasks

    def _make_decision(
        self,
        tasks: List[ExecutiveTask],
        requires_approval: bool
    ) -> ExecutiveDecision:
        """
        Make executive decision on whether to proceed with tasks

        Args:
            tasks: List of executive tasks
            requires_approval: Whether approval is required

        Returns:
            ExecutiveDecision
        """
        # Check resource availability
        if len(self.active_tasks) + len(tasks) > self.max_concurrent_tasks:
            logger.warning(f"Resource limit exceeded: {len(self.active_tasks)}/{self.max_concurrent_tasks}")
            return ExecutiveDecision.DEFER

        # Check for high-priority tasks
        critical_tasks = [t for t in tasks if t.priority >= 4]
        if critical_tasks and requires_approval:
            logger.info(f"Escalating {len(critical_tasks)} critical tasks for approval")
            return ExecutiveDecision.ESCALATE

        # Approve by default
        return ExecutiveDecision.APPROVED

    async def _create_routing_plan(
        self,
        executive_tasks: List[ExecutiveTask]
    ) -> Dict[ClusterType, List[TaskContext]]:
        """
        Create routing plan for executive tasks

        Args:
            executive_tasks: List of executive tasks

        Returns:
            Dict mapping cluster types to task contexts
        """
        routing_plan: Dict[ClusterType, List[TaskContext]] = {
            cluster: [] for cluster in ClusterType
        }

        for exec_task in executive_tasks:
            # Convert executive task to agent-level task contexts
            task_context = self._executive_to_task_context(exec_task)
            routing_plan[exec_task.cluster].append(task_context)

        # Log routing plan
        for cluster, tasks in routing_plan.items():
            if tasks:
                logger.info(f"Cluster {cluster.value}: {len(tasks)} tasks")

        return routing_plan

    def _executive_to_task_context(self, exec_task: ExecutiveTask) -> TaskContext:
        """
        Convert executive task to agent-level task context

        Args:
            exec_task: Executive task

        Returns:
            TaskContext for agent routing
        """
        # Map cluster to task type (simplified for now)
        task_type_mapping = {
            ClusterType.CREATIVE: TaskType.IMAGE_GENERATION,
            ClusterType.COMMERCE: TaskType.WORDPRESS_THEME,
            ClusterType.BRAND_AI: TaskType.DATA_ANALYSIS,
            ClusterType.DATA_LAYER: TaskType.KPI_DASHBOARD,
            ClusterType.INFRASTRUCTURE: TaskType.CODE_AUDIT
        }

        return TaskContext(
            task_type=task_type_mapping.get(exec_task.cluster, TaskType.DATA_ANALYSIS),
            description=exec_task.description,
            priority=exec_task.priority,
            metadata=exec_task.metadata
        )

    async def _execute_with_orchestration(
        self,
        routing_plan: Dict[ClusterType, List[TaskContext]]
    ) -> List[Dict[str, Any]]:
        """
        Execute tasks with orchestration and resource management

        Args:
            routing_plan: Routing plan with tasks per cluster

        Returns:
            List of execution results
        """
        results = []

        # Execute tasks by cluster (could be parallelized in production)
        for cluster, tasks in routing_plan.items():
            if not tasks:
                continue

            logger.info(f"Executing {len(tasks)} tasks in {cluster.value} cluster")

            for task in tasks:
                # Route task to appropriate agent
                decision = self.agent_router.route_task(task)

                # Execute task (simulated for now - would call actual agents in production)
                result = {
                    'cluster': cluster.value,
                    'task': task.description,
                    'agent': decision.agent_name,
                    'command': decision.orchestration_command,
                    'confidence': decision.confidence,
                    'success': True,  # Simulated success
                    'timestamp': datetime.now().isoformat()
                }

                results.append(result)
                logger.info(f"Task routed to {decision.agent_name} ({decision.confidence:.0%} confidence)")

        return results

    def _create_report(
        self,
        start_time: datetime,
        executed: int,
        successful: int
    ) -> ExecutiveReport:
        """
        Create executive summary report

        Args:
            start_time: Mission start time
            executed: Number of tasks executed
            successful: Number of successful tasks

        Returns:
            ExecutiveReport
        """
        duration = (datetime.now() - start_time).total_seconds()

        # Update performance metrics
        self.performance_metrics['total_tasks'] += executed
        self.performance_metrics['successful_tasks'] += successful
        self.performance_metrics['failed_tasks'] += (executed - successful)

        return ExecutiveReport(
            timestamp=datetime.now(),
            tasks_executed=executed,
            tasks_successful=successful,
            tasks_failed=executed - successful,
            clusters_activated=[],  # Would track actual cluster usage
            resource_utilization={'cpu': 0.45, 'memory': 0.62},  # Simulated
            performance_metrics={
                'execution_time_seconds': duration,
                'tasks_per_second': executed / duration if duration > 0 else 0
            },
            recommendations=self._generate_recommendations(successful, executed)
        )

    def _generate_recommendations(
        self,
        successful: int,
        total: int
    ) -> List[str]:
        """
        Generate executive recommendations based on performance

        Args:
            successful: Number of successful tasks
            total: Total number of tasks

        Returns:
            List of recommendations
        """
        recommendations = []
        success_rate = (successful / total * 100) if total > 0 else 0

        if success_rate < 80:
            recommendations.append("Warning: Success rate below 80%. Review agent configurations.")

        if success_rate == 100:
            recommendations.append("Excellent performance. Consider increasing task complexity.")

        if total > self.max_concurrent_tasks * 0.8:
            recommendations.append("Approaching resource limits. Consider scaling infrastructure.")

        return recommendations

    async def get_system_health(self) -> Dict[str, Any]:
        """
        Get comprehensive system health status

        Returns:
            Dict with system health metrics
        """
        agents = self.agent_loader.load_all_agents()

        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'agents': {
                'total': len(agents),
                'by_type': {}
            },
            'performance': self.performance_metrics,
            'active_tasks': len(self.active_tasks),
            'task_history': len(self.task_history)
        }

        # Count agents by type
        for agent in agents.values():
            agent_type = agent.agent_type.value
            health['agents']['by_type'][agent_type] = health['agents']['by_type'].get(agent_type, 0) + 1

        return health


# Example usage and testing
async def main():
    """Example executive orchestrator usage"""
    print("\n" + "=" * 70)
    print("Executive Orchestrator (GOD MODE) - Example Usage")
    print("=" * 70 + "\n")

    # Initialize orchestrator
    executive = ExecutiveOrchestrator()
    await executive.initialize()

    # Example missions
    missions = [
        {
            'description': "Create luxury brand landing page with product images and marketing copy",
            'priority': 4
        },
        {
            'description': "Analyze customer data and build revenue dashboard with KPIs",
            'priority': 3
        },
        {
            'description': "Generate high-quality product photography for e-commerce site",
            'priority': 5
        }
    ]

    # Execute missions
    for i, mission in enumerate(missions, 1):
        print(f"\n### Mission {i}: {mission['description'][:50]}...")
        report = await executive.execute_mission(
            mission['description'],
            priority=mission['priority']
        )

        print(f"Executed: {report.tasks_executed}")
        print(f"Successful: {report.tasks_successful}")
        print(f"Failed: {report.tasks_failed}")
        print(f"Performance: {report.performance_metrics}")

        if report.recommendations:
            print(f"Recommendations:")
            for rec in report.recommendations:
                print(f"  - {rec}")

    # System health
    print("\n### System Health")
    health = await executive.get_system_health()
    print(json.dumps(health, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
