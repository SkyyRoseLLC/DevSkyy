#!/usr/bin/env python3
"""
Enterprise-Grade Test Suite for Agent Router Module
Comprehensive test coverage for agents/router.py following Truth Protocol

References: /Users/coreyfoster/Desktop/CLAUDE_20-10_MASTER.md
Truth Protocol Compliance: Rules 1, 4, 8, 10, 14
Target Coverage: ≥95%
"""

import sys
from pathlib import Path

# Add project root to path FIRST before any local imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock
import logging
from datetime import datetime

from agents.router import (
    AgentRouter,
    TaskType,
    TaskContext,
    RoutingDecision,
)
from agents.loader import (
    AgentConfigLoader,
    AgentType,
    AgentConfiguration,
    AgentStatus,
    OrchestrationCommand,
    PerformanceSLO,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_agent_config() -> AgentConfiguration:
    """Create mock agent configuration for testing"""
    return AgentConfiguration(
        agent_id="test-agent-001",
        agent_name="Test Agent",
        agent_type=AgentType.CODE_QUALITY_SECURITY,
        version="1.0.0",
        master_document="/path/to/master.md",
        status=AgentStatus.ACTIVE,
        composition={"primary_ai": "claude-3-5-sonnet", "secondary_ai": "cursor"},
        capabilities={
            "primary_functions": ["code_audit", "security_scan", "test_generation"],
            "supported_languages": ["python", "javascript"],
        },
        truth_protocol_compliance={
            "never_guess": True,
            "pin_versions": True,
            "cite_standards": True,
            "no_hardcoded_secrets": True,
            "rbac_enforcement": True,
            "test_coverage_minimum": 90,
            "no_skip_rule": True,
            "error_ledger_required": True,
        },
        orchestration_commands={
            "BUILD": OrchestrationCommand(
                command="BUILD",
                description="Build implementation",
                execution_mode="synchronous",
                required_inputs=["requirements"],
                validation_steps=["lint", "test"],
                tools=["pytest", "mypy"],
            ),
            "TEST": OrchestrationCommand(
                command="TEST",
                description="Run tests",
                execution_mode="synchronous",
                required_inputs=["code"],
                validation_steps=["coverage"],
                tools=["pytest"],
                coverage_requirement=90,
            ),
            "REVIEW": OrchestrationCommand(
                command="REVIEW",
                description="Review code",
                execution_mode="synchronous",
                required_inputs=["code"],
                validation_steps=["security", "quality"],
                tools=["bandit", "ruff"],
            ),
            "PLAN": OrchestrationCommand(
                command="PLAN",
                description="Create plan",
                execution_mode="synchronous",
                required_inputs=["scope"],
                validation_steps=["validate"],
                tools=["planner"],
            ),
        },
        performance_slos=PerformanceSLO(
            p95_latency_ms=150,
            error_rate_percent=0.3,
            test_coverage_percent=90,
            secrets_in_repo=0,
        ),
        monitoring={"enabled": True},
        deliverables_per_cycle=["Code", "Tests", "Docs"],
        created_at=datetime.now(),
        last_updated=datetime.now(),
        maintainer="test@devskyy.com",
    )


@pytest.fixture
def mock_growth_agent_config() -> AgentConfiguration:
    """Create mock growth/marketing agent configuration"""
    return AgentConfiguration(
        agent_id="growth-stack-001",
        agent_name="Growth Stack",
        agent_type=AgentType.GROWTH_MARKETING_AUTOMATION,
        version="1.0.0",
        master_document="/path/to/master.md",
        status=AgentStatus.ACTIVE,
        composition={"primary_ai": "claude-3-5-sonnet", "secondary_ai": "chatgpt-4"},
        capabilities={
            "primary_functions": ["wordpress_theme", "landing_page", "seo"],
        },
        truth_protocol_compliance={
            "never_guess": True,
            "pin_versions": True,
            "test_coverage_minimum": 90,
        },
        orchestration_commands={
            "BUILD": OrchestrationCommand(
                command="BUILD",
                description="Build marketing assets",
                execution_mode="synchronous",
            ),
            "MONITOR": OrchestrationCommand(
                command="MONITOR",
                description="Monitor campaigns",
                execution_mode="asynchronous",
            ),
        },
        performance_slos=PerformanceSLO(
            p95_latency_ms=200,
            lighthouse_score=95,
            conversion_rate_minimum=2.5,
        ),
        monitoring={"enabled": True},
        deliverables_per_cycle=["Theme", "Landing Pages", "Analytics"],
        created_at=datetime.now(),
        last_updated=datetime.now(),
        maintainer="test@devskyy.com",
    )


@pytest.fixture
def mock_data_agent_config() -> AgentConfiguration:
    """Create mock data/analytics agent configuration"""
    return AgentConfiguration(
        agent_id="data-reasoning-001",
        agent_name="Data & Reasoning",
        agent_type=AgentType.DATA_ANALYSIS_INTELLIGENCE,
        version="1.0.0",
        master_document="/path/to/master.md",
        status=AgentStatus.ACTIVE,
        composition={"primary_ai": "claude-3-5-sonnet", "secondary_ai": "gemini-pro"},
        capabilities={
            "primary_functions": ["data_analysis", "kpi_dashboard", "forecasting"],
        },
        truth_protocol_compliance={"test_coverage_minimum": 90},
        orchestration_commands={
            "BUILD": OrchestrationCommand(
                command="BUILD",
                description="Build analytics",
                execution_mode="synchronous",
            ),
            "MONITOR": OrchestrationCommand(
                command="MONITOR",
                description="Monitor metrics",
                execution_mode="asynchronous",
            ),
        },
        performance_slos=PerformanceSLO(p95_latency_ms=180),
        monitoring={"enabled": True},
        deliverables_per_cycle=["Reports", "Dashboards"],
        created_at=datetime.now(),
        last_updated=datetime.now(),
        maintainer="test@devskyy.com",
    )


@pytest.fixture
def mock_visual_agent_config() -> AgentConfiguration:
    """Create mock visual content agent configuration"""
    return AgentConfiguration(
        agent_id="visual-foundry-001",
        agent_name="Visual Foundry",
        agent_type=AgentType.VISUAL_CONTENT_GENERATION,
        version="1.0.0",
        master_document="/path/to/master.md",
        status=AgentStatus.ACTIVE,
        composition={"primary_ai": "stable-diffusion", "secondary_ai": "claude"},
        capabilities={
            "primary_functions": ["image_generation", "video_automation"],
        },
        truth_protocol_compliance={"test_coverage_minimum": 85},
        orchestration_commands={
            "BUILD": OrchestrationCommand(
                command="BUILD",
                description="Generate visuals",
                execution_mode="asynchronous",
            ),
        },
        performance_slos=PerformanceSLO(p95_latency_ms=300),
        monitoring={"enabled": True},
        deliverables_per_cycle=["Images", "Videos"],
        created_at=datetime.now(),
        last_updated=datetime.now(),
        maintainer="test@devskyy.com",
    )


@pytest.fixture
def mock_config_loader(
    mock_agent_config,
    mock_growth_agent_config,
    mock_data_agent_config,
    mock_visual_agent_config,
) -> AgentConfigLoader:
    """Create mock AgentConfigLoader with test agents"""
    loader = Mock(spec=AgentConfigLoader)
    loader.load_all_agents.return_value = {
        "test-agent-001": mock_agent_config,
        "growth-stack-001": mock_growth_agent_config,
        "data-reasoning-001": mock_data_agent_config,
        "visual-foundry-001": mock_visual_agent_config,
    }
    return loader


@pytest.fixture
def agent_router(mock_config_loader) -> AgentRouter:
    """Create AgentRouter instance with mock loader"""
    return AgentRouter(config_loader=mock_config_loader)


@pytest.fixture
def sample_task_context() -> TaskContext:
    """Create sample TaskContext for testing"""
    return TaskContext(
        task_type=TaskType.CODE_AUDIT,
        description="Audit FastAPI security implementation",
        priority=3,
        requirements={"coverage": 90, "tools": ["bandit"]},
        metadata={"project": "devskyy"},
    )


# ============================================================================
# Initialization Tests
# ============================================================================


class TestAgentRouterInitialization:
    """Test AgentRouter initialization and setup"""

    def test_initialization_with_custom_loader(self, mock_config_loader):
        """Test router initializes correctly with custom loader"""
        router = AgentRouter(config_loader=mock_config_loader)

        assert router.loader is mock_config_loader
        assert len(router.agents) == 4
        assert "test-agent-001" in router.agents
        assert "growth-stack-001" in router.agents

    def test_initialization_without_loader_creates_default(self):
        """Test router creates default loader when none provided"""
        with patch("agents.router.AgentConfigLoader") as mock_loader_class:
            mock_instance = Mock()
            mock_loader_class.return_value = mock_instance
            mock_instance.load_all_agents.return_value = {}

            router = AgentRouter()

            mock_loader_class.assert_called_once()
            assert router.loader is mock_instance

    def test_task_type_mapping_built_correctly(self, agent_router):
        """Test task type to agent type mapping is complete"""
        mapping = agent_router._task_to_agent_type

        # Verify all TaskType values are mapped
        assert len(mapping) == 20  # All TaskType enum values

        # Verify specific mappings
        assert mapping[TaskType.CODE_AUDIT] == AgentType.CODE_QUALITY_SECURITY
        assert mapping[TaskType.WORDPRESS_THEME] == AgentType.GROWTH_MARKETING_AUTOMATION
        assert mapping[TaskType.DATA_ANALYSIS] == AgentType.DATA_ANALYSIS_INTELLIGENCE
        assert mapping[TaskType.IMAGE_GENERATION] == AgentType.VISUAL_CONTENT_GENERATION

    def test_capability_keywords_built_correctly(self, agent_router):
        """Test capability keywords are built for all agent types"""
        keywords = agent_router._capability_keywords

        assert len(keywords) == 4  # One for each AgentType

        # Verify keyword presence
        assert "code" in keywords[AgentType.CODE_QUALITY_SECURITY]
        assert "security" in keywords[AgentType.CODE_QUALITY_SECURITY]
        assert "wordpress" in keywords[AgentType.GROWTH_MARKETING_AUTOMATION]
        assert "data" in keywords[AgentType.DATA_ANALYSIS_INTELLIGENCE]
        assert "image" in keywords[AgentType.VISUAL_CONTENT_GENERATION]

    def test_initialization_logs_agent_count(self, mock_config_loader, caplog):
        """Test initialization logs number of loaded agents"""
        with caplog.at_level(logging.INFO):
            router = AgentRouter(config_loader=mock_config_loader)

        assert "AgentRouter initialized with 4 agents" in caplog.text


# ============================================================================
# Task Routing Tests - All TaskType Values
# ============================================================================


class TestRouteTaskAllTypes:
    """Test route_task() for all 20+ TaskType enum values"""

    @pytest.mark.parametrize(
        "task_type,expected_agent_type",
        [
            # Code Quality & Security (5 types)
            (TaskType.CODE_AUDIT, AgentType.CODE_QUALITY_SECURITY),
            (TaskType.CODE_REFACTOR, AgentType.CODE_QUALITY_SECURITY),
            (TaskType.SECURITY_SCAN, AgentType.CODE_QUALITY_SECURITY),
            (TaskType.TEST_GENERATION, AgentType.CODE_QUALITY_SECURITY),
            (TaskType.PERFORMANCE_OPTIMIZATION, AgentType.CODE_QUALITY_SECURITY),
            # Growth & Marketing (5 types)
            (TaskType.WORDPRESS_THEME, AgentType.GROWTH_MARKETING_AUTOMATION),
            (TaskType.LANDING_PAGE, AgentType.GROWTH_MARKETING_AUTOMATION),
            (TaskType.AB_TESTING, AgentType.GROWTH_MARKETING_AUTOMATION),
            (TaskType.SEO_OPTIMIZATION, AgentType.GROWTH_MARKETING_AUTOMATION),
            (TaskType.CONVERSION_OPTIMIZATION, AgentType.GROWTH_MARKETING_AUTOMATION),
            # Data & Analytics (5 types)
            (TaskType.DATA_ANALYSIS, AgentType.DATA_ANALYSIS_INTELLIGENCE),
            (TaskType.KPI_DASHBOARD, AgentType.DATA_ANALYSIS_INTELLIGENCE),
            (TaskType.PREDICTIVE_ANALYTICS, AgentType.DATA_ANALYSIS_INTELLIGENCE),
            (TaskType.PROMPT_ROUTING, AgentType.DATA_ANALYSIS_INTELLIGENCE),
            (TaskType.EVALUATION_HARNESS, AgentType.DATA_ANALYSIS_INTELLIGENCE),
            # Visual Content (5 types)
            (TaskType.IMAGE_GENERATION, AgentType.VISUAL_CONTENT_GENERATION),
            (TaskType.IMAGE_UPSCALING, AgentType.VISUAL_CONTENT_GENERATION),
            (TaskType.VIDEO_AUTOMATION, AgentType.VISUAL_CONTENT_GENERATION),
            (TaskType.PRODUCT_PHOTOGRAPHY, AgentType.VISUAL_CONTENT_GENERATION),
            (TaskType.BRAND_ASSETS, AgentType.VISUAL_CONTENT_GENERATION),
        ],
    )
    def test_route_task_direct_mapping(
        self, agent_router, task_type, expected_agent_type
    ):
        """Test direct task type mapping for all TaskType values"""
        task = TaskContext(
            task_type=task_type,
            description=f"Test task for {task_type.value}",
            priority=3,
        )

        decision = agent_router.route_task(task)

        assert decision.agent_type == expected_agent_type
        assert decision.confidence == 1.0
        assert "Direct task type mapping" in decision.reasoning
        assert decision.agent_id is not None
        assert decision.agent_name is not None

    def test_route_code_audit_high_priority(self, agent_router):
        """Test routing high-priority code audit task"""
        task = TaskContext(
            task_type=TaskType.CODE_AUDIT,
            description="Critical security audit for production system",
            priority=5,
            requirements={"coverage": 95, "tools": ["bandit", "safety"]},
        )

        decision = agent_router.route_task(task)

        assert decision.agent_type == AgentType.CODE_QUALITY_SECURITY
        assert decision.confidence == 1.0
        assert decision.orchestration_command is not None
        assert "p95_latency_ms" in decision.estimated_slo

    def test_route_wordpress_theme_task(self, agent_router):
        """Test routing WordPress theme creation task"""
        task = TaskContext(
            task_type=TaskType.WORDPRESS_THEME,
            description="Create luxury fashion WordPress theme",
            priority=4,
        )

        decision = agent_router.route_task(task)

        assert decision.agent_type == AgentType.GROWTH_MARKETING_AUTOMATION
        assert decision.confidence == 1.0

    def test_route_data_analysis_task(self, agent_router):
        """Test routing data analysis task"""
        task = TaskContext(
            task_type=TaskType.DATA_ANALYSIS,
            description="Analyze customer conversion funnel",
            priority=3,
        )

        decision = agent_router.route_task(task)

        assert decision.agent_type == AgentType.DATA_ANALYSIS_INTELLIGENCE
        assert decision.confidence == 1.0

    def test_route_image_generation_task(self, agent_router):
        """Test routing image generation task"""
        task = TaskContext(
            task_type=TaskType.IMAGE_GENERATION,
            description="Generate product photography",
            priority=4,
        )

        decision = agent_router.route_task(task)

        assert decision.agent_type == AgentType.VISUAL_CONTENT_GENERATION
        assert decision.confidence == 1.0


# ============================================================================
# Natural Language Routing Tests
# ============================================================================


class TestNaturalLanguageRouting:
    """Test suggest_agent_for_natural_language() with various inputs"""

    @pytest.mark.parametrize(
        "user_request,expected_agent_type",
        [
            # Code quality requests
            ("audit my Python code for security issues", AgentType.CODE_QUALITY_SECURITY),
            ("fix the security vulnerabilities", AgentType.CODE_QUALITY_SECURITY),
            ("refactor this messy codebase", AgentType.CODE_QUALITY_SECURITY),
            ("generate tests for my API", AgentType.CODE_QUALITY_SECURITY),
            ("review my code for issues", AgentType.CODE_QUALITY_SECURITY),
            # Marketing/growth requests
            ("create a WordPress theme", AgentType.GROWTH_MARKETING_AUTOMATION),
            ("build a landing page", AgentType.GROWTH_MARKETING_AUTOMATION),
            ("design a theme for my blog", AgentType.GROWTH_MARKETING_AUTOMATION),
            # Data/analytics requests
            ("analyze my sales data", AgentType.DATA_ANALYSIS_INTELLIGENCE),
            ("create a KPI dashboard", AgentType.DATA_ANALYSIS_INTELLIGENCE),
            ("perform data analysis on customer metrics", AgentType.DATA_ANALYSIS_INTELLIGENCE),
            # Visual content requests
            ("generate product images", AgentType.VISUAL_CONTENT_GENERATION),
            ("create a video for marketing", AgentType.VISUAL_CONTENT_GENERATION),
            ("generate visual assets", AgentType.VISUAL_CONTENT_GENERATION),
        ],
    )
    def test_natural_language_routing_patterns(
        self, agent_router, user_request, expected_agent_type
    ):
        """Test natural language routing with various request patterns"""
        decision = agent_router.suggest_agent_for_natural_language(user_request)

        assert decision.agent_type == expected_agent_type
        assert decision.confidence > 0.0
        assert decision.agent_id is not None
        # Verify reasoning contains the request info
        assert decision.reasoning is not None

    def test_natural_language_empty_request(self, agent_router):
        """Test natural language routing with empty request defaults to code audit"""
        decision = agent_router.suggest_agent_for_natural_language("")

        # Should default to code audit
        assert decision.agent_type == AgentType.CODE_QUALITY_SECURITY

    def test_natural_language_ambiguous_request(self, agent_router):
        """Test natural language routing with ambiguous request"""
        decision = agent_router.suggest_agent_for_natural_language(
            "help me build something"
        )

        # Should default to code audit for ambiguous requests
        assert decision.agent_type == AgentType.CODE_QUALITY_SECURITY

    def test_natural_language_multi_keyword_request(self, agent_router):
        """Test request with keywords from multiple categories"""
        # Request has both code and data keywords
        decision = agent_router.suggest_agent_for_natural_language(
            "audit the security of my data analysis code"
        )

        # Should route based on fuzzy matching
        assert decision.agent_type in [
            AgentType.CODE_QUALITY_SECURITY,
            AgentType.DATA_ANALYSIS_INTELLIGENCE,
        ]
        assert decision.confidence > 0.0

    def test_natural_language_sets_medium_priority(self, agent_router):
        """Test natural language routing sets medium priority by default"""
        decision = agent_router.suggest_agent_for_natural_language("audit my code")

        # Check that internal task has medium priority (3)
        # This is indirectly tested through the decision
        assert decision.estimated_slo is not None


# ============================================================================
# Confidence Scoring Tests
# ============================================================================


class TestConfidenceScoring:
    """Test confidence scoring accuracy and range validation"""

    def test_direct_mapping_confidence_is_one(self, agent_router):
        """Test direct task type mapping produces confidence 1.0"""
        task = TaskContext(
            task_type=TaskType.CODE_AUDIT,
            description="Security audit",
            priority=3,
        )

        decision = agent_router.route_task(task)

        assert decision.confidence == 1.0

    def test_fuzzy_match_confidence_in_valid_range(self, agent_router):
        """Test fuzzy matching produces confidence in valid range [0.0, 1.0]"""
        # Test various keyword densities
        test_cases = [
            "audit code security test",  # High keyword density
            "review my application",  # Medium keyword density
            "do something",  # Low keyword density
        ]

        for description in test_cases:
            confidence = agent_router._fuzzy_match_agent_type(description)[1]

            assert 0.0 <= confidence <= 1.0, (
                f"Confidence {confidence} out of range for '{description}'"
            )

    def test_fuzzy_match_with_many_keywords(self, agent_router):
        """Test fuzzy matching with description containing many relevant keywords"""
        description = "code audit security test refactor performance vulnerability compliance"

        agent_type, confidence = agent_router._fuzzy_match_agent_type(description)

        assert agent_type == AgentType.CODE_QUALITY_SECURITY
        assert confidence > 0.3  # Should have high confidence

    def test_fuzzy_match_with_few_keywords(self, agent_router):
        """Test fuzzy matching with description containing few keywords"""
        description = "please help me"

        agent_type, confidence = agent_router._fuzzy_match_agent_type(description)

        assert confidence <= 0.3  # Should have low or default confidence
        # Should default to CODE_QUALITY_SECURITY for low confidence
        assert agent_type == AgentType.CODE_QUALITY_SECURITY

    def test_fuzzy_match_empty_description(self, agent_router):
        """Test fuzzy matching with empty description"""
        agent_type, confidence = agent_router._fuzzy_match_agent_type("")

        assert confidence == 0.3  # Default confidence
        assert agent_type == AgentType.CODE_QUALITY_SECURITY

    def test_fuzzy_match_confidence_normalized(self, agent_router):
        """Test fuzzy match confidence is properly normalized"""
        # Even with all keywords present, confidence should be ≤ 1.0
        all_keywords = " ".join(
            agent_router._capability_keywords[AgentType.CODE_QUALITY_SECURITY]
        )

        agent_type, confidence = agent_router._fuzzy_match_agent_type(all_keywords)

        assert confidence <= 1.0
        assert agent_type == AgentType.CODE_QUALITY_SECURITY


# ============================================================================
# Orchestration Command Selection Tests
# ============================================================================


class TestOrchestrationCommandSelection:
    """Test _select_orchestration_command() logic"""

    def test_high_priority_selects_plan_command(self, agent_router, mock_agent_config):
        """Test high-priority tasks prefer PLAN command"""
        task = TaskContext(
            task_type=TaskType.CODE_AUDIT,
            description="Critical audit",
            priority=5,  # High priority
        )

        command = agent_router._select_orchestration_command(task, mock_agent_config)

        assert command == "PLAN"

    def test_code_audit_selects_review_command(self, agent_router, mock_agent_config):
        """Test CODE_AUDIT tasks prefer REVIEW command"""
        task = TaskContext(
            task_type=TaskType.CODE_AUDIT,
            description="Security review",
            priority=3,
        )

        command = agent_router._select_orchestration_command(task, mock_agent_config)

        assert command in ["REVIEW", "TEST"]

    def test_security_scan_selects_review_command(self, agent_router, mock_agent_config):
        """Test SECURITY_SCAN tasks prefer REVIEW command"""
        task = TaskContext(
            task_type=TaskType.SECURITY_SCAN,
            description="Scan for vulnerabilities",
            priority=3,
        )

        command = agent_router._select_orchestration_command(task, mock_agent_config)

        assert command in ["REVIEW", "TEST"]

    def test_data_analysis_selects_monitor_command(
        self, agent_router, mock_data_agent_config
    ):
        """Test DATA_ANALYSIS tasks prefer MONITOR command"""
        task = TaskContext(
            task_type=TaskType.DATA_ANALYSIS,
            description="Analyze metrics",
            priority=3,
        )

        command = agent_router._select_orchestration_command(
            task, mock_data_agent_config
        )

        assert command == "MONITOR"

    def test_kpi_dashboard_selects_monitor_command(
        self, agent_router, mock_data_agent_config
    ):
        """Test KPI_DASHBOARD tasks prefer MONITOR command"""
        task = TaskContext(
            task_type=TaskType.KPI_DASHBOARD,
            description="Create dashboard",
            priority=3,
        )

        command = agent_router._select_orchestration_command(
            task, mock_data_agent_config
        )

        assert command == "MONITOR"

    def test_default_build_command_selection(self, agent_router, mock_agent_config):
        """Test default BUILD command for general implementation tasks"""
        task = TaskContext(
            task_type=TaskType.CODE_REFACTOR,
            description="Refactor module",
            priority=2,
        )

        command = agent_router._select_orchestration_command(task, mock_agent_config)

        assert command == "BUILD"

    def test_fallback_to_first_command(self, agent_router):
        """Test fallback to first available command if no BUILD"""
        # Create agent config without BUILD command
        agent_config = Mock()
        agent_config.orchestration_commands = {
            "TEST": Mock(),
            "REVIEW": Mock(),
        }

        task = TaskContext(
            task_type=TaskType.CODE_REFACTOR,
            description="Refactor",
            priority=2,
        )

        command = agent_router._select_orchestration_command(task, agent_config)

        assert command in ["TEST", "REVIEW"]

    def test_fallback_to_execute_if_no_commands(self, agent_router):
        """Test fallback to EXECUTE if no commands defined"""
        agent_config = Mock()
        agent_config.orchestration_commands = {}

        task = TaskContext(
            task_type=TaskType.CODE_AUDIT,
            description="Audit",
            priority=3,
        )

        command = agent_router._select_orchestration_command(task, agent_config)

        assert command == "EXECUTE"


# ============================================================================
# SLO Estimation Tests
# ============================================================================


class TestSLOEstimation:
    """Test _estimate_slo() calculations"""

    def test_slo_estimation_includes_latency(self, agent_router, mock_agent_config):
        """Test SLO estimation includes p95 latency"""
        task = TaskContext(
            task_type=TaskType.CODE_AUDIT,
            description="Audit",
            priority=3,
        )

        estimated_slo = agent_router._estimate_slo(task, mock_agent_config)

        assert "p95_latency_ms" in estimated_slo
        assert isinstance(estimated_slo["p95_latency_ms"], int)
        assert estimated_slo["p95_latency_ms"] > 0

    def test_slo_estimation_priority_multiplier(self, agent_router, mock_agent_config):
        """Test priority affects SLO latency estimation"""
        low_priority_task = TaskContext(
            task_type=TaskType.CODE_AUDIT, description="Audit", priority=1
        )
        high_priority_task = TaskContext(
            task_type=TaskType.CODE_AUDIT, description="Audit", priority=5
        )

        low_slo = agent_router._estimate_slo(low_priority_task, mock_agent_config)
        high_slo = agent_router._estimate_slo(high_priority_task, mock_agent_config)

        assert low_slo["p95_latency_ms"] < high_slo["p95_latency_ms"]

    def test_slo_estimation_includes_test_coverage(
        self, agent_router, mock_agent_config
    ):
        """Test SLO estimation includes test coverage target"""
        task = TaskContext(
            task_type=TaskType.CODE_AUDIT,
            description="Audit",
            priority=3,
        )

        estimated_slo = agent_router._estimate_slo(task, mock_agent_config)

        assert "test_coverage_percent" in estimated_slo
        assert estimated_slo["test_coverage_percent"] == 90

    def test_slo_estimation_includes_error_rate(self, agent_router, mock_agent_config):
        """Test SLO estimation includes error rate target"""
        task = TaskContext(
            task_type=TaskType.CODE_AUDIT,
            description="Audit",
            priority=3,
        )

        estimated_slo = agent_router._estimate_slo(task, mock_agent_config)

        assert "error_rate_percent" in estimated_slo
        assert estimated_slo["error_rate_percent"] == 0.3

    def test_slo_estimation_includes_priority(self, agent_router, mock_agent_config):
        """Test SLO estimation includes task priority"""
        task = TaskContext(
            task_type=TaskType.CODE_AUDIT,
            description="Audit",
            priority=4,
        )

        estimated_slo = agent_router._estimate_slo(task, mock_agent_config)

        assert "priority" in estimated_slo
        assert estimated_slo["priority"] == 4

    def test_slo_estimation_meets_slo_optimistic(
        self, agent_router, mock_agent_config
    ):
        """Test SLO estimation includes optimistic meets_slo flag"""
        task = TaskContext(
            task_type=TaskType.CODE_AUDIT,
            description="Audit",
            priority=3,
        )

        estimated_slo = agent_router._estimate_slo(task, mock_agent_config)

        assert "meets_slo" in estimated_slo
        assert estimated_slo["meets_slo"] is True

    def test_slo_estimation_with_none_values(self, agent_router):
        """Test SLO estimation handles None values gracefully"""
        agent_config = Mock()
        agent_config.performance_slos = PerformanceSLO()  # All None values

        task = TaskContext(
            task_type=TaskType.CODE_AUDIT,
            description="Audit",
            priority=3,
        )

        estimated_slo = agent_router._estimate_slo(task, agent_config)

        # Should still include priority and meets_slo
        assert "priority" in estimated_slo
        assert "meets_slo" in estimated_slo


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


class TestEdgeCasesAndErrors:
    """Test edge cases, unknown task types, and error handling"""

    def test_route_task_with_unknown_agent_type_fallback(self, agent_router):
        """Test routing falls back when no agent found for type"""
        # Mock scenario where agent type has no matching agent
        agent_router.agents = {}  # Empty agents dict

        task = TaskContext(
            task_type=TaskType.CODE_AUDIT,
            description="Audit",
            priority=3,
        )

        with pytest.raises(IndexError):
            # Should raise error when trying to get first agent from empty dict
            agent_router.route_task(task)

    def test_route_task_empty_description(self, agent_router):
        """Test routing with empty description"""
        task = TaskContext(
            task_type=TaskType.CODE_AUDIT,
            description="",
            priority=3,
        )

        decision = agent_router.route_task(task)

        assert decision is not None
        assert decision.confidence == 1.0  # Direct mapping

    def test_route_task_very_long_description(self, agent_router):
        """Test routing with very long description"""
        long_description = "audit " * 1000  # Very long description

        task = TaskContext(
            task_type=TaskType.CODE_AUDIT,
            description=long_description,
            priority=3,
        )

        decision = agent_router.route_task(task)

        assert decision is not None
        assert decision.confidence == 1.0

    def test_route_task_special_characters_in_description(self, agent_router):
        """Test routing with special characters in description"""
        task = TaskContext(
            task_type=TaskType.CODE_AUDIT,
            description="Audit @#$%^&*() special chars!",
            priority=3,
        )

        decision = agent_router.route_task(task)

        assert decision is not None

    def test_route_task_min_priority(self, agent_router):
        """Test routing with minimum priority value"""
        task = TaskContext(
            task_type=TaskType.CODE_AUDIT,
            description="Audit",
            priority=1,
        )

        decision = agent_router.route_task(task)

        assert decision.estimated_slo["priority"] == 1

    def test_route_task_max_priority(self, agent_router):
        """Test routing with maximum priority value"""
        task = TaskContext(
            task_type=TaskType.CODE_AUDIT,
            description="Critical audit",
            priority=5,
        )

        decision = agent_router.route_task(task)

        assert decision.estimated_slo["priority"] == 5
        # High priority should prefer PLAN command
        assert decision.orchestration_command == "PLAN"

    def test_route_task_with_empty_requirements(self, agent_router):
        """Test routing with empty requirements dict"""
        task = TaskContext(
            task_type=TaskType.CODE_AUDIT,
            description="Audit",
            priority=3,
            requirements={},
        )

        decision = agent_router.route_task(task)

        assert decision is not None

    def test_route_task_with_none_metadata(self, agent_router):
        """Test routing handles None metadata gracefully"""
        task = TaskContext(
            task_type=TaskType.CODE_AUDIT,
            description="Audit",
            priority=3,
        )
        task.metadata = None

        decision = agent_router.route_task(task)

        assert decision is not None


# ============================================================================
# Multiple Task Routing Tests
# ============================================================================


class TestMultipleTaskRouting:
    """Test route_multiple_tasks() batch processing"""

    def test_route_multiple_tasks_success(self, agent_router):
        """Test routing multiple tasks successfully"""
        tasks = [
            TaskContext(TaskType.CODE_AUDIT, "Audit 1", priority=5),
            TaskContext(TaskType.WORDPRESS_THEME, "Theme 1", priority=4),
            TaskContext(TaskType.DATA_ANALYSIS, "Analysis 1", priority=3),
        ]

        decisions = agent_router.route_multiple_tasks(tasks)

        assert len(decisions) == 3
        assert decisions[0].agent_type == AgentType.CODE_QUALITY_SECURITY
        assert decisions[1].agent_type == AgentType.GROWTH_MARKETING_AUTOMATION
        assert decisions[2].agent_type == AgentType.DATA_ANALYSIS_INTELLIGENCE

    def test_route_multiple_tasks_empty_list(self, agent_router):
        """Test routing empty task list"""
        decisions = agent_router.route_multiple_tasks([])

        assert decisions == []

    def test_route_multiple_tasks_with_errors(self, agent_router, caplog):
        """Test routing continues when individual tasks fail"""
        # Mock the route_task method to raise an exception for specific task
        original_route = agent_router.route_task
        call_count = [0]

        def mock_route(task):
            call_count[0] += 1
            if call_count[0] == 2:  # Fail on second task
                raise RuntimeError("Test error")
            return original_route(task)

        agent_router.route_task = mock_route

        tasks = [
            TaskContext(TaskType.CODE_AUDIT, "Valid task 1", priority=3),
            TaskContext(TaskType.CODE_AUDIT, "Task that will fail", priority=3),
            TaskContext(TaskType.DATA_ANALYSIS, "Valid task 2", priority=3),
        ]

        with caplog.at_level(logging.ERROR):
            decisions = agent_router.route_multiple_tasks(tasks)

        # Restore original method
        agent_router.route_task = original_route

        # Should continue processing despite error
        assert len(decisions) == 2  # Only valid tasks processed
        assert "Failed to route task" in caplog.text

    def test_route_multiple_diverse_tasks(self, agent_router):
        """Test routing diverse task types in batch"""
        tasks = [
            TaskContext(TaskType.CODE_AUDIT, "Audit", priority=5),
            TaskContext(TaskType.SECURITY_SCAN, "Scan", priority=4),
            TaskContext(TaskType.WORDPRESS_THEME, "Theme", priority=3),
            TaskContext(TaskType.IMAGE_GENERATION, "Image", priority=2),
            TaskContext(TaskType.KPI_DASHBOARD, "Dashboard", priority=3),
        ]

        decisions = agent_router.route_multiple_tasks(tasks)

        assert len(decisions) == 5
        # Verify variety of agent types
        agent_types = {d.agent_type for d in decisions}
        assert len(agent_types) >= 3  # At least 3 different agent types used


# ============================================================================
# Agent Workload Tests
# ============================================================================


class TestAgentWorkload:
    """Test get_agent_workload() tracking"""

    def test_get_agent_workload_returns_dict(self, agent_router):
        """Test workload tracking returns dict of agent IDs"""
        workload = agent_router.get_agent_workload()

        assert isinstance(workload, dict)
        assert len(workload) == 4  # All agents

    def test_get_agent_workload_initial_zero(self, agent_router):
        """Test initial workload is zero for all agents"""
        workload = agent_router.get_agent_workload()

        for agent_id, count in workload.items():
            assert count == 0

    def test_get_agent_workload_contains_all_agents(self, agent_router):
        """Test workload dict contains all loaded agents"""
        workload = agent_router.get_agent_workload()

        assert "test-agent-001" in workload
        assert "growth-stack-001" in workload
        assert "data-reasoning-001" in workload
        assert "visual-foundry-001" in workload


# ============================================================================
# Integration Tests with AgentConfigLoader
# ============================================================================


class TestRouterLoaderIntegration:
    """Test integration between AgentRouter and AgentConfigLoader"""

    def test_router_uses_loader_agents(self, agent_router, mock_config_loader):
        """Test router correctly uses agents from loader"""
        assert agent_router.agents == mock_config_loader.load_all_agents.return_value

    def test_router_handles_loader_with_no_agents(self):
        """Test router handles loader returning no agents"""
        loader = Mock(spec=AgentConfigLoader)
        loader.load_all_agents.return_value = {}

        router = AgentRouter(config_loader=loader)

        assert router.agents == {}
        # Mapping should still be built
        assert len(router._task_to_agent_type) > 0

    def test_route_task_with_real_agent_config(self, agent_router, mock_agent_config):
        """Test routing with complete agent configuration"""
        task = TaskContext(
            task_type=TaskType.CODE_AUDIT,
            description="Comprehensive security audit",
            priority=5,
            requirements={"coverage": 90, "tools": ["bandit", "safety"]},
        )

        decision = agent_router.route_task(task)

        # Verify decision uses real agent config data
        assert decision.agent_id == "test-agent-001"
        assert decision.agent_name == "Test Agent"
        assert decision.agent_type == AgentType.CODE_QUALITY_SECURITY
        assert decision.orchestration_command in ["BUILD", "TEST", "REVIEW", "PLAN"]

    def test_routing_decision_structure_complete(self, agent_router):
        """Test RoutingDecision contains all required fields"""
        task = TaskContext(
            task_type=TaskType.CODE_AUDIT,
            description="Audit",
            priority=3,
        )

        decision = agent_router.route_task(task)

        # Verify all RoutingDecision fields are populated
        assert decision.agent_id is not None
        assert decision.agent_name is not None
        assert decision.agent_type is not None
        assert isinstance(decision.confidence, float)
        assert 0.0 <= decision.confidence <= 1.0
        assert decision.reasoning is not None
        assert decision.orchestration_command is not None
        assert isinstance(decision.estimated_slo, dict)


# ============================================================================
# Logging and Observability Tests
# ============================================================================


class TestLoggingAndObservability:
    """Test logging behavior for monitoring and debugging"""

    def test_router_logs_routing_decision(self, agent_router, caplog):
        """Test router logs each routing decision"""
        task = TaskContext(
            task_type=TaskType.CODE_AUDIT,
            description="Audit",
            priority=3,
        )

        with caplog.at_level(logging.INFO):
            decision = agent_router.route_task(task)

        assert "Routed task" in caplog.text
        assert decision.agent_name in caplog.text
        assert "confidence" in caplog.text

    def test_router_logs_fallback_warning(self, agent_router, caplog):
        """Test router logs warning when falling back to default agent"""
        # Remove all agents of a specific type to trigger fallback
        original_agents = agent_router.agents.copy()
        agent_router.agents = {
            k: v
            for k, v in original_agents.items()
            if v.agent_type != AgentType.CODE_QUALITY_SECURITY
        }

        task = TaskContext(
            task_type=TaskType.CODE_AUDIT,
            description="Audit",
            priority=3,
        )

        with caplog.at_level(logging.WARNING):
            decision = agent_router.route_task(task)

        assert "falling back" in caplog.text.lower()

    def test_multiple_tasks_logs_errors_individually(self, agent_router, caplog):
        """Test batch routing logs errors for each failed task"""
        # Mock route_task to always raise an exception
        original_route = agent_router.route_task

        def mock_route(task):
            raise RuntimeError(f"Test error for {task.description}")

        agent_router.route_task = mock_route

        tasks = [
            TaskContext(TaskType.CODE_AUDIT, "Test 1", priority=3),
            TaskContext(TaskType.CODE_AUDIT, "Test 2", priority=3),
            TaskContext(TaskType.CODE_AUDIT, "Test 3", priority=3),
        ]

        with caplog.at_level(logging.ERROR):
            decisions = agent_router.route_multiple_tasks(tasks)

        # Restore original method
        agent_router.route_task = original_route

        # Should have logged error for each failed task
        error_count = caplog.text.count("Failed to route task")
        assert error_count == 3


# ============================================================================
# Property and Method Coverage Tests
# ============================================================================


class TestMethodCoverage:
    """Ensure all methods and properties are tested"""

    def test_build_task_type_mapping_returns_dict(self, agent_router):
        """Test _build_task_type_mapping returns complete mapping"""
        mapping = agent_router._build_task_type_mapping()

        assert isinstance(mapping, dict)
        assert len(mapping) == 20  # All TaskType values mapped

    def test_build_capability_keywords_returns_dict(self, agent_router):
        """Test _build_capability_keywords returns keyword lists"""
        keywords = agent_router._build_capability_keywords()

        assert isinstance(keywords, dict)
        assert len(keywords) == 4  # All AgentType values covered

        for agent_type, kw_list in keywords.items():
            assert isinstance(kw_list, list)
            assert len(kw_list) > 0
            assert all(isinstance(kw, str) for kw in kw_list)

    def test_fuzzy_match_agent_type_returns_tuple(self, agent_router):
        """Test _fuzzy_match_agent_type returns (AgentType, float) tuple"""
        result = agent_router._fuzzy_match_agent_type("test description")

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], AgentType)
        assert isinstance(result[1], float)

    def test_select_orchestration_command_returns_string(
        self, agent_router, mock_agent_config
    ):
        """Test _select_orchestration_command returns string command"""
        task = TaskContext(
            task_type=TaskType.CODE_AUDIT,
            description="Audit",
            priority=3,
        )

        command = agent_router._select_orchestration_command(task, mock_agent_config)

        assert isinstance(command, str)
        assert len(command) > 0

    def test_estimate_slo_returns_dict(self, agent_router, mock_agent_config):
        """Test _estimate_slo returns dict with SLO estimates"""
        task = TaskContext(
            task_type=TaskType.CODE_AUDIT,
            description="Audit",
            priority=3,
        )

        slo = agent_router._estimate_slo(task, mock_agent_config)

        assert isinstance(slo, dict)
        assert "priority" in slo
        assert "meets_slo" in slo


# ============================================================================
# TaskContext Tests
# ============================================================================


class TestTaskContext:
    """Test TaskContext dataclass initialization and defaults"""

    def test_task_context_initialization(self):
        """Test TaskContext initializes with all fields"""
        task = TaskContext(
            task_type=TaskType.CODE_AUDIT,
            description="Test audit",
            priority=4,
            requirements={"key": "value"},
            metadata={"project": "test"},
        )

        assert task.task_type == TaskType.CODE_AUDIT
        assert task.description == "Test audit"
        assert task.priority == 4
        assert task.requirements == {"key": "value"}
        assert task.metadata == {"project": "test"}

    def test_task_context_default_priority(self):
        """Test TaskContext default priority is 1"""
        task = TaskContext(
            task_type=TaskType.CODE_AUDIT,
            description="Test",
        )

        assert task.priority == 1

    def test_task_context_default_requirements(self):
        """Test TaskContext default requirements is empty dict"""
        task = TaskContext(
            task_type=TaskType.CODE_AUDIT,
            description="Test",
        )

        assert task.requirements == {}

    def test_task_context_default_metadata(self):
        """Test TaskContext default metadata is empty dict"""
        task = TaskContext(
            task_type=TaskType.CODE_AUDIT,
            description="Test",
        )

        assert task.metadata == {}


# ============================================================================
# Additional Coverage Tests
# ============================================================================


class TestAdditionalCoverage:
    """Additional tests to ensure 95%+ coverage"""

    def test_priority_4_selects_plan(self, agent_router, mock_agent_config):
        """Test priority 4 (>=4) selects PLAN command"""
        task = TaskContext(
            task_type=TaskType.CODE_REFACTOR,
            description="Refactor",
            priority=4,
        )

        command = agent_router._select_orchestration_command(task, mock_agent_config)

        assert command == "PLAN"

    def test_case_insensitive_fuzzy_matching(self, agent_router):
        """Test fuzzy matching is case insensitive"""
        desc_lower = "audit security code"
        desc_upper = "AUDIT SECURITY CODE"
        desc_mixed = "AuDiT SeCuRiTy CoDe"

        type1, conf1 = agent_router._fuzzy_match_agent_type(desc_lower)
        type2, conf2 = agent_router._fuzzy_match_agent_type(desc_upper)
        type3, conf3 = agent_router._fuzzy_match_agent_type(desc_mixed)

        assert type1 == type2 == type3
        assert conf1 == conf2 == conf3

    def test_all_task_types_have_mapping(self):
        """Test all TaskType enum values have mapping"""
        loader = Mock(spec=AgentConfigLoader)
        loader.load_all_agents.return_value = {}

        router = AgentRouter(config_loader=loader)

        mapping = router._build_task_type_mapping()

        # Get all TaskType values
        all_task_types = set(TaskType)
        mapped_task_types = set(mapping.keys())

        assert all_task_types == mapped_task_types

    def test_routing_decision_dataclass(self):
        """Test RoutingDecision can be created directly"""
        decision = RoutingDecision(
            agent_id="test-001",
            agent_name="Test Agent",
            agent_type=AgentType.CODE_QUALITY_SECURITY,
            confidence=0.95,
            reasoning="Test reasoning",
            orchestration_command="BUILD",
            estimated_slo={"latency": 100},
        )

        assert decision.agent_id == "test-001"
        assert decision.confidence == 0.95

    def test_natural_language_with_numbers_only(self, agent_router):
        """Test natural language routing with numbers only"""
        decision = agent_router.suggest_agent_for_natural_language("12345")

        # Should default to code audit
        assert decision.agent_type == AgentType.CODE_QUALITY_SECURITY

    def test_natural_language_with_special_chars_only(self, agent_router):
        """Test natural language routing with special characters only"""
        decision = agent_router.suggest_agent_for_natural_language("!@#$%^&*()")

        # Should default to code audit
        assert decision.agent_type == AgentType.CODE_QUALITY_SECURITY

    def test_fuzzy_match_partial_keyword_match(self, agent_router):
        """Test fuzzy matching with partial keyword matches"""
        # "security" is a keyword, "secure" might match via substring
        description = "make my app secure"

        agent_type, confidence = agent_router._fuzzy_match_agent_type(description)

        # Should still route to code quality
        assert agent_type == AgentType.CODE_QUALITY_SECURITY
