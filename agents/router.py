#!/usr/bin/env python3
"""
Agent Task Router with Confidence Scoring
Routes tasks to appropriate agents using exact, fuzzy, and fallback strategies

Truth Protocol Compliance: CLAUDE.md
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

from agents.loader import AgentConfigLoader, AgentConfig, AgentConfigError

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Task type enumeration for routing (30 types)"""
    # Security & Scanning
    SECURITY_SCAN = "security_scan"
    VULNERABILITY_SCAN = "vulnerability_scan"
    COMPLIANCE_CHECK = "compliance_check"
    PENETRATION_TEST = "penetration_test"

    # Code Generation & Fixing
    CODE_GENERATION = "code_generation"
    CODE_REFACTOR = "code_refactor"
    CODE_FIX = "code_fix"
    AUTO_FIX = "auto_fix"
    CODE_REVIEW = "code_review"

    # Machine Learning
    ML_TRAINING = "ml_training"
    ML_INFERENCE = "ml_inference"
    MODEL_OPTIMIZATION = "model_optimization"
    FEATURE_ENGINEERING = "feature_engineering"

    # Testing
    TEST_GENERATION = "test_generation"
    INTEGRATION_TEST = "integration_test"
    PERFORMANCE_TEST = "performance_test"

    # Documentation
    DOCUMENTATION_GENERATION = "documentation_generation"
    API_DOCUMENTATION = "api_documentation"

    # Database
    DATABASE_OPTIMIZATION = "database_optimization"
    SCHEMA_MIGRATION = "schema_migration"

    # Deployment
    DEPLOYMENT = "deployment"
    CONTAINER_BUILD = "container_build"
    CI_CD_PIPELINE = "ci_cd_pipeline"

    # Monitoring & Analytics
    PERFORMANCE_MONITORING = "performance_monitoring"
    ERROR_TRACKING = "error_tracking"
    LOG_ANALYSIS = "log_analysis"

    # API & Integration
    API_DESIGN = "api_design"
    INTEGRATION_DEVELOPMENT = "integration_development"

    # General
    GENERAL_TASK = "general_task"
    CUSTOM_TASK = "custom_task"


@dataclass
class TaskRequest:
    """Task request with metadata"""
    task_type: TaskType
    description: str
    priority: int = 50
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not 1 <= self.priority <= 100:
            raise ValueError(f"Priority must be between 1 and 100, got {self.priority}")


@dataclass
class TaskResult:
    """Task routing result with confidence and reasoning"""
    agent_name: str
    confidence: float
    reasoning: str
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")


@dataclass
class ScoringWeights:
    """Weights for agent scoring algorithm"""
    priority_alignment: float = 0.40
    capability_confidence: float = 0.40
    availability: float = 0.20

    def __post_init__(self):
        total = self.priority_alignment + self.capability_confidence + self.availability
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


class AgentRouter:
    """
    Routes tasks to appropriate agents with confidence scoring

    Features:
    - Exact routing: Direct name/ID match (0.95 confidence)
    - Fuzzy routing: Keyword similarity + NLP (0.60-0.85 confidence)
    - Fallback routing: General agent (0.30 confidence)
    - Batch task processing for MCP efficiency
    - Agent scoring: Priority (40%), Capability (40%), Availability (20%)

    Usage:
        router = AgentRouter()
        task = TaskRequest(
            task_type=TaskType.SECURITY_SCAN,
            description="Scan Python code for vulnerabilities",
            priority=80
        )
        result = router.route(task)
        print(f"Routed to: {result.agent_name} (confidence: {result.confidence})")
    """

    def __init__(
        self,
        config_loader: Optional[AgentConfigLoader] = None,
        scoring_weights: Optional[ScoringWeights] = None
    ):
        """
        Initialize agent router

        Args:
            config_loader: AgentConfigLoader instance (creates new if None)
            scoring_weights: Custom scoring weights (uses defaults if None)
        """
        self.loader = config_loader or AgentConfigLoader()
        self.weights = scoring_weights or ScoringWeights()

        # Load all available agents
        self.agents: Dict[str, AgentConfig] = {}
        try:
            self.agents = self.loader.load_all()
        except AgentConfigError as e:
            logger.warning(f"Failed to load agents: {e}")

        # Build routing tables
        self._capability_keywords = self._build_capability_keywords()
        self._task_type_mapping = self._build_task_type_mapping()

        logger.info(
            f"AgentRouter initialized with {len(self.agents)} agents, "
            f"weights: P={self.weights.priority_alignment:.0%}, "
            f"C={self.weights.capability_confidence:.0%}, "
            f"A={self.weights.availability:.0%}"
        )

    def _build_capability_keywords(self) -> Dict[str, List[str]]:
        """
        Build keyword mapping for fuzzy matching

        Returns:
            Dict mapping capability to list of relevant keywords
        """
        return {
            'security_scan': [
                'security', 'vulnerability', 'scan', 'audit', 'penetration',
                'exploit', 'cve', 'owasp', 'injection', 'xss', 'csrf'
            ],
            'code_fix': [
                'fix', 'repair', 'patch', 'debug', 'resolve', 'correct',
                'autofix', 'heal', 'remediate'
            ],
            'code_generation': [
                'generate', 'create', 'build', 'scaffold', 'template',
                'boilerplate', 'code', 'implement'
            ],
            'ml_training': [
                'train', 'machine learning', 'ml', 'model', 'neural',
                'deep learning', 'ai', 'learning', 'dataset'
            ],
            'test_generation': [
                'test', 'unittest', 'pytest', 'coverage', 'assertion',
                'mock', 'integration test'
            ],
            'performance_optimization': [
                'performance', 'optimize', 'speed', 'latency', 'throughput',
                'bottleneck', 'profiling', 'benchmark'
            ],
            'documentation': [
                'document', 'docs', 'readme', 'comment', 'docstring',
                'api docs', 'guide', 'tutorial'
            ],
            'database': [
                'database', 'sql', 'query', 'schema', 'migration',
                'index', 'postgres', 'mysql'
            ],
            'deployment': [
                'deploy', 'deployment', 'docker', 'kubernetes', 'container',
                'ci/cd', 'pipeline', 'release'
            ],
            'monitoring': [
                'monitor', 'metrics', 'logging', 'observability', 'tracing',
                'alert', 'dashboard', 'prometheus'
            ],
            'api_design': [
                'api', 'rest', 'graphql', 'endpoint', 'route', 'openapi',
                'swagger', 'interface'
            ]
        }

    def _build_task_type_mapping(self) -> Dict[TaskType, List[str]]:
        """
        Build mapping from task types to required capabilities

        Returns:
            Dict mapping TaskType to list of capabilities
        """
        return {
            # Security
            TaskType.SECURITY_SCAN: ['security_scan', 'vulnerability_scan'],
            TaskType.VULNERABILITY_SCAN: ['security_scan', 'vulnerability_scan'],
            TaskType.COMPLIANCE_CHECK: ['security_scan', 'compliance'],
            TaskType.PENETRATION_TEST: ['security_scan', 'penetration_test'],

            # Code
            TaskType.CODE_GENERATION: ['code_generation', 'code_generation'],
            TaskType.CODE_REFACTOR: ['code_refactor', 'code_generation'],
            TaskType.CODE_FIX: ['code_fix', 'auto_fix'],
            TaskType.AUTO_FIX: ['auto_fix', 'code_fix'],
            TaskType.CODE_REVIEW: ['code_review', 'security_scan'],

            # ML
            TaskType.ML_TRAINING: ['ml_training', 'ml_inference'],
            TaskType.ML_INFERENCE: ['ml_inference', 'ml_training'],
            TaskType.MODEL_OPTIMIZATION: ['model_optimization', 'ml_training'],
            TaskType.FEATURE_ENGINEERING: ['feature_engineering', 'ml_training'],

            # Testing
            TaskType.TEST_GENERATION: ['test_generation', 'code_generation'],
            TaskType.INTEGRATION_TEST: ['integration_test', 'test_generation'],
            TaskType.PERFORMANCE_TEST: ['performance_test', 'monitoring'],

            # Documentation
            TaskType.DOCUMENTATION_GENERATION: ['documentation', 'code_generation'],
            TaskType.API_DOCUMENTATION: ['api_documentation', 'documentation'],

            # Database
            TaskType.DATABASE_OPTIMIZATION: ['database', 'performance_optimization'],
            TaskType.SCHEMA_MIGRATION: ['database', 'migration'],

            # Deployment
            TaskType.DEPLOYMENT: ['deployment', 'ci_cd'],
            TaskType.CONTAINER_BUILD: ['container_build', 'deployment'],
            TaskType.CI_CD_PIPELINE: ['ci_cd', 'deployment'],

            # Monitoring
            TaskType.PERFORMANCE_MONITORING: ['monitoring', 'performance_optimization'],
            TaskType.ERROR_TRACKING: ['error_tracking', 'monitoring'],
            TaskType.LOG_ANALYSIS: ['log_analysis', 'monitoring'],

            # API
            TaskType.API_DESIGN: ['api_design', 'code_generation'],
            TaskType.INTEGRATION_DEVELOPMENT: ['integration', 'api_design'],

            # General
            TaskType.GENERAL_TASK: ['general'],
            TaskType.CUSTOM_TASK: ['general'],
        }

    def route_exact(self, agent_name_or_id: str) -> Optional[TaskResult]:
        """
        Route to specific agent by exact name or ID match

        Args:
            agent_name_or_id: Agent name or ID to route to

        Returns:
            TaskResult if agent found, None otherwise

        Note:
            Returns 0.95 confidence for exact matches
        """
        # Try by ID first
        if agent_name_or_id in self.agents:
            agent = self.agents[agent_name_or_id]
            return TaskResult(
                agent_name=agent.name,
                confidence=0.95,
                reasoning=f"Exact agent ID match: {agent_name_or_id}",
                metadata={'agent_id': agent.agent_id, 'routing_method': 'exact'}
            )

        # Try by name (case-insensitive)
        name_lower = agent_name_or_id.lower()
        for agent_id, agent in self.agents.items():
            if agent.name.lower() == name_lower:
                return TaskResult(
                    agent_name=agent.name,
                    confidence=0.95,
                    reasoning=f"Exact agent name match: {agent.name}",
                    metadata={'agent_id': agent.agent_id, 'routing_method': 'exact'}
                )

        logger.debug(f"No exact match found for: {agent_name_or_id}")
        return None

    def route_fuzzy(self, task: TaskRequest) -> Optional[TaskResult]:
        """
        Route task using fuzzy matching with keyword similarity

        Args:
            task: TaskRequest to route

        Returns:
            TaskResult with confidence 0.60-0.85 based on match quality

        Note:
            Uses NLP-style keyword matching and agent scoring algorithm
        """
        if not self.agents:
            logger.warning("No agents available for fuzzy routing")
            return None

        # Get required capabilities for task type
        required_capabilities = self._task_type_mapping.get(task.task_type, ['general'])

        # Find agents with matching capabilities
        candidate_agents = []
        for agent_id, agent in self.agents.items():
            if agent.has_any_capability(required_capabilities):
                candidate_agents.append(agent)

        if not candidate_agents:
            # Fallback: try keyword matching on description
            candidate_agents = self._keyword_match_agents(task.description)

        if not candidate_agents:
            logger.debug(f"No candidates found for fuzzy routing: {task.task_type.value}")
            return None

        # Score candidates
        best_agent = None
        best_score = 0.0

        for agent in candidate_agents:
            score = self._score_agent(agent, task)
            if score > best_score:
                best_score = score
                best_agent = agent

        if best_agent is None:
            return None

        # Map score to confidence (0.60-0.85 range)
        confidence = 0.60 + (best_score * 0.25)

        return TaskResult(
            agent_name=best_agent.name,
            confidence=confidence,
            reasoning=(
                f"Fuzzy match for {task.task_type.value}: "
                f"capability overlap, priority alignment, score={best_score:.2f}"
            ),
            metadata={
                'agent_id': best_agent.agent_id,
                'routing_method': 'fuzzy',
                'raw_score': best_score
            }
        )

    def route_fallback(self) -> Optional[TaskResult]:
        """
        Fallback routing to general-purpose agent

        Returns:
            TaskResult with confidence 0.30 for fallback routing

        Note:
            Selects highest-priority available agent as fallback
        """
        if not self.agents:
            logger.error("No agents available for fallback routing")
            return None

        # Get available agents
        available = self.loader.get_available_agents(list(self.agents.values()))

        if not available:
            # No available agents, use any agent
            available = list(self.agents.values())

        # Sort by priority (descending)
        available.sort(key=lambda a: a.priority, reverse=True)
        fallback_agent = available[0]

        return TaskResult(
            agent_name=fallback_agent.name,
            confidence=0.30,
            reasoning=(
                f"Fallback routing to highest-priority agent: "
                f"{fallback_agent.name} (priority={fallback_agent.priority})"
            ),
            metadata={
                'agent_id': fallback_agent.agent_id,
                'routing_method': 'fallback'
            }
        )

    def route(self, task: TaskRequest, prefer_exact: Optional[str] = None) -> TaskResult:
        """
        Route task using all available strategies

        Args:
            task: TaskRequest to route
            prefer_exact: Optional agent name/ID for exact routing preference

        Returns:
            TaskResult with routing decision

        Note:
            Routing order: exact -> fuzzy -> fallback
        """
        # Try exact routing if preferred agent specified
        if prefer_exact:
            exact_result = self.route_exact(prefer_exact)
            if exact_result:
                logger.info(
                    f"Routed task via exact match to {exact_result.agent_name} "
                    f"(confidence: {exact_result.confidence:.2f})"
                )
                return exact_result

        # Try fuzzy routing
        fuzzy_result = self.route_fuzzy(task)
        if fuzzy_result and fuzzy_result.confidence >= 0.60:
            logger.info(
                f"Routed task via fuzzy match to {fuzzy_result.agent_name} "
                f"(confidence: {fuzzy_result.confidence:.2f})"
            )
            return fuzzy_result

        # Fallback routing
        fallback_result = self.route_fallback()
        if fallback_result:
            logger.warning(
                f"Routed task via fallback to {fallback_result.agent_name} "
                f"(confidence: {fallback_result.confidence:.2f})"
            )
            return fallback_result

        # Ultimate fallback: return error result
        logger.error("All routing strategies failed")
        raise AgentConfigError("No agents available for routing")

    def route_multiple_tasks(
        self,
        tasks: List[TaskRequest]
    ) -> List[TaskResult]:
        """
        Route multiple tasks efficiently (MCP batch operation)

        Args:
            tasks: List of TaskRequest objects to route

        Returns:
            List of TaskResult objects (same order as input)

        Note:
            This is an MCP efficiency pattern - routes multiple tasks
            in a single operation to reduce overhead
        """
        results = []
        errors = []

        for i, task in enumerate(tasks):
            try:
                result = self.route(task)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to route task {i}: {e}")
                errors.append((i, str(e)))

        if errors:
            logger.warning(f"Batch routing completed with {len(errors)} errors")

        logger.info(f"Batch routed {len(results)}/{len(tasks)} tasks")
        return results

    def _score_agent(self, agent: AgentConfig, task: TaskRequest) -> float:
        """
        Score agent for task using weighted algorithm

        Args:
            agent: AgentConfig to score
            task: TaskRequest to score against

        Returns:
            Score between 0.0 and 1.0

        Algorithm:
            - Priority alignment: 40% (normalized priority difference)
            - Capability confidence: 40% (capability match quality)
            - Availability: 20% (agent availability status)
        """
        # Priority alignment (40%)
        priority_diff = abs(agent.priority - task.priority)
        priority_score = 1.0 - (priority_diff / 100.0)
        priority_score = max(0.0, min(1.0, priority_score))

        # Capability confidence (40%)
        required_caps = self._task_type_mapping.get(task.task_type, ['general'])
        capability_score = self._calculate_capability_score(agent, required_caps, task.description)

        # Availability (20%)
        availability_score = 1.0 if agent.available else 0.3

        # Weighted total
        total_score = (
            priority_score * self.weights.priority_alignment +
            capability_score * self.weights.capability_confidence +
            availability_score * self.weights.availability
        )

        logger.debug(
            f"Score for {agent.name}: total={total_score:.2f} "
            f"(P={priority_score:.2f}, C={capability_score:.2f}, A={availability_score:.2f})"
        )

        return total_score

    def _calculate_capability_score(
        self,
        agent: AgentConfig,
        required_capabilities: List[str],
        description: str
    ) -> float:
        """
        Calculate capability match score

        Args:
            agent: AgentConfig to score
            required_capabilities: Required capabilities for task
            description: Task description for keyword matching

        Returns:
            Score between 0.0 and 1.0
        """
        # Direct capability match (60% weight)
        direct_matches = sum(
            1 for cap in required_capabilities
            if agent.matches_capability(cap)
        )
        direct_score = direct_matches / len(required_capabilities) if required_capabilities else 0.0

        # Keyword match in description (40% weight)
        keyword_score = self._calculate_keyword_score(agent, description)

        return (direct_score * 0.6) + (keyword_score * 0.4)

    def _calculate_keyword_score(self, agent: AgentConfig, description: str) -> float:
        """
        Calculate keyword match score between agent capabilities and description

        Args:
            agent: AgentConfig to score
            description: Task description

        Returns:
            Score between 0.0 and 1.0
        """
        description_lower = description.lower()

        # Get keywords for agent's capabilities
        relevant_keywords = []
        for capability in agent.capabilities:
            cap_lower = capability.lower()
            if cap_lower in self._capability_keywords:
                relevant_keywords.extend(self._capability_keywords[cap_lower])

        if not relevant_keywords:
            return 0.0

        # Count keyword matches
        matches = sum(1 for keyword in relevant_keywords if keyword in description_lower)

        return min(1.0, matches / 5.0)  # Normalize to 1.0 with 5+ matches

    def _keyword_match_agents(self, description: str) -> List[AgentConfig]:
        """
        Find agents matching keywords in description

        Args:
            description: Task description

        Returns:
            List of AgentConfig objects with keyword matches
        """
        description_lower = description.lower()
        matching_agents = []

        for agent_id, agent in self.agents.items():
            for capability in agent.capabilities:
                cap_lower = capability.lower()
                if cap_lower in self._capability_keywords:
                    keywords = self._capability_keywords[cap_lower]
                    if any(kw in description_lower for kw in keywords):
                        matching_agents.append(agent)
                        break

        return matching_agents

    def get_routing_stats(self) -> Dict[str, Any]:
        """
        Get routing statistics

        Returns:
            Dictionary with routing statistics
        """
        available_agents = self.loader.get_available_agents(list(self.agents.values()))

        return {
            'total_agents': len(self.agents),
            'available_agents': len(available_agents),
            'unavailable_agents': len(self.agents) - len(available_agents),
            'scoring_weights': {
                'priority_alignment': self.weights.priority_alignment,
                'capability_confidence': self.weights.capability_confidence,
                'availability': self.weights.availability
            },
            'task_types_supported': len(TaskType),
            'capability_keywords': len(self._capability_keywords)
        }

    def explain_routing(self, task: TaskRequest) -> Dict[str, Any]:
        """
        Explain routing decision for a task (debugging tool)

        Args:
            task: TaskRequest to explain

        Returns:
            Dictionary with detailed routing explanation
        """
        explanation = {
            'task_type': task.task_type.value,
            'task_priority': task.priority,
            'required_capabilities': self._task_type_mapping.get(task.task_type, ['general']),
            'candidates': []
        }

        # Score all agents
        for agent_id, agent in self.agents.items():
            score = self._score_agent(agent, task)
            explanation['candidates'].append({
                'agent_id': agent_id,
                'agent_name': agent.name,
                'score': score,
                'priority': agent.priority,
                'available': agent.available,
                'capabilities': agent.capabilities
            })

        # Sort by score
        explanation['candidates'].sort(key=lambda x: x['score'], reverse=True)

        # Add final routing decision
        result = self.route(task)
        explanation['final_decision'] = {
            'agent_name': result.agent_name,
            'confidence': result.confidence,
            'reasoning': result.reasoning,
            'metadata': result.metadata
        }

        return explanation

    def refresh_agents(self) -> None:
        """
        Refresh agent configurations from loader

        Useful when agent configs are updated at runtime
        """
        try:
            self.loader.refresh_configs()
            self.agents = self.loader.load_all()
            logger.info(f"Refreshed agents: {len(self.agents)} agents loaded")
        except AgentConfigError as e:
            logger.error(f"Failed to refresh agents: {e}")
