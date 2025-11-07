#!/usr/bin/env python3
"""
Comprehensive Test Suite for AgentConfigLoader
Enterprise-grade pytest tests with ≥90% code coverage

Truth Protocol Compliance: Rules 1, 2, 3, 8, 10, 15
Coverage Target: ≥90%
Test Categories:
  - Initialization tests
  - load_agent() tests (success, file not found, invalid JSON, missing fields)
  - load_all_agents() tests
  - validate_agent() tests (all Truth Protocol rules)
  - get_agent_by_type() tests
  - get_quick_reference() tests
  - Cache behavior tests
  - Error handling tests
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import logging
import pytest
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, patch, mock_open, MagicMock

from agents.loader import (
    AgentConfigLoader,
    AgentConfiguration,
    AgentType,
    AgentStatus,
    OrchestrationCommand,
    PerformanceSLO
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def config_dir(tmp_path):
    """Create temporary config directory for testing"""
    config_path = tmp_path / "config"
    config_path.mkdir()
    return config_path


@pytest.fixture
def valid_agent_index():
    """Valid agent index JSON data"""
    return {
        "index_version": "2.0.0",
        "master_document": "/path/to/master.md",
        "created_at": "2025-11-04T16:00:00Z",
        "last_updated": "2025-11-04T16:00:00Z",
        "total_agents": 2,
        "agents": [
            {
                "agent_id": "test-agent-001",
                "agent_name": "Test Agent",
                "agent_type": "code_quality_security",
                "config_file": "/tmp/test_agent.json",
                "status": "active",
                "priority": 1
            },
            {
                "agent_id": "test-agent-002",
                "agent_name": "Test Agent 2",
                "agent_type": "growth_marketing_automation",
                "config_file": "/tmp/test_agent_2.json",
                "status": "inactive",
                "priority": 2
            }
        ],
        "quick_reference": {
            "code_quality": "test-agent-001",
            "marketing_growth": "test-agent-002"
        }
    }


@pytest.fixture
def valid_agent_config():
    """Valid agent configuration JSON data"""
    return {
        "agent_id": "test-agent-001",
        "agent_name": "Test Agent",
        "agent_type": "code_quality_security",
        "version": "2.0.0",
        "master_document": "/path/to/master.md",
        "status": "active",
        "composition": {
            "primary_ai": "Claude Sonnet 4.5",
            "secondary_ai": "ChatGPT-5-Codex",
            "collaboration_mode": "sequential"
        },
        "capabilities": {
            "primary_functions": ["Backend code audits", "Security enforcement"],
            "technical_stack": {
                "languages": ["Python 3.11.9"],
                "frameworks": ["FastAPI 0.104"]
            }
        },
        "truth_protocol_compliance": {
            "never_guess": True,
            "pin_versions": True,
            "cite_standards": ["RFC 7519", "NIST SP 800-38D"],
            "no_hardcoded_secrets": True,
            "rbac_enforcement": True,
            "test_coverage_minimum": 90,
            "no_skip_rule": True,
            "error_ledger_required": True
        },
        "orchestration_commands": {
            "PLAN": {
                "command": "PLAN(scope)",
                "description": "Create job graph",
                "execution_mode": "synchronous",
                "required_inputs": ["scope"],
                "validation_steps": ["validate_scope"],
                "tools": ["planner"],
                "coverage_requirement": 90
            },
            "BUILD": {
                "command": "BUILD(job_id)",
                "description": "Execute implementation",
                "execution_mode": "asynchronous",
                "required_inputs": ["job_id"],
                "validation_steps": [],
                "tools": ["builder"]
            }
        },
        "performance_slos": {
            "p95_latency_ms": 200,
            "error_rate_percent": 0.5,
            "test_coverage_percent": 90,
            "secrets_in_repo": 0,
            "page_load_time_ms": 1000,
            "lighthouse_score": 95,
            "conversion_rate_minimum": 0.02
        },
        "monitoring": {
            "endpoints": {
                "metrics": "/api/v1/monitoring/metrics",
                "health": "/api/v1/healthz"
            }
        },
        "deliverables_per_cycle": [
            "Code + Docs + Tests",
            "OpenAPI + Coverage + SBOM"
        ],
        "created_at": "2025-11-04T16:00:00Z",
        "last_updated": "2025-11-04T16:00:00Z",
        "maintainer": "DevSkyy Platform Team",
        "wordpress_integration": {
            "enabled": True,
            "version": "6.x"
        },
        "ci_cd_integration": {
            "platform": "GitHub Actions",
            "workflow_file": ".github/workflows/ci.yml"
        }
    }


@pytest.fixture
def minimal_agent_config():
    """Minimal valid agent configuration (required fields only)"""
    return {
        "agent_id": "minimal-agent-001",
        "agent_name": "Minimal Agent",
        "agent_type": "data_analysis_intelligence",
        "version": "1.0.0",
        "master_document": "/path/to/master.md",
        "status": "maintenance",
        "composition": {},
        "capabilities": {},
        "truth_protocol_compliance": {},
        "orchestration_commands": {},
        "performance_slos": {},
        "monitoring": {},
        "deliverables_per_cycle": [],
        "created_at": "2025-11-04T16:00:00Z",
        "last_updated": "2025-11-04T16:00:00Z",
        "maintainer": "Test Team"
    }


@pytest.fixture
def invalid_json_data():
    """Invalid JSON string for testing"""
    return '{"agent_id": "test", "broken": True, invalid json}'


@pytest.fixture
def agent_config_missing_required_fields():
    """Agent config missing required fields"""
    return {
        "agent_id": "incomplete-agent",
        "agent_name": "Incomplete Agent"
        # Missing: agent_type, version, status, etc.
    }


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

class TestAgentConfigLoaderInitialization:
    """Test AgentConfigLoader initialization and setup"""

    def test_init_with_default_config_dir(self, tmp_path, monkeypatch):
        """Test initialization with default config directory"""
        # Create config dir next to loader.py
        loader_parent = Path(__file__).parent.parent.parent / "agents"
        config_path = loader_parent / "config"

        # Ensure config directory exists
        assert config_path.exists(), "Default config directory should exist"

        loader = AgentConfigLoader()
        assert loader.config_dir == config_path
        assert loader.index_path == config_path / "agents_index.json"
        assert loader._index is None
        assert loader._agents_cache == {}

    def test_init_with_custom_config_dir(self, config_dir):
        """Test initialization with custom config directory"""
        loader = AgentConfigLoader(config_dir=config_dir)
        assert loader.config_dir == config_dir
        assert loader.index_path == config_dir / "agents_index.json"
        assert loader._index is None
        assert loader._agents_cache == {}

    def test_init_with_nonexistent_config_dir(self, tmp_path):
        """Test initialization fails with non-existent config directory"""
        nonexistent_path = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError) as exc_info:
            AgentConfigLoader(config_dir=nonexistent_path)

        assert "Config directory not found" in str(exc_info.value)
        assert str(nonexistent_path) in str(exc_info.value)

    def test_init_with_string_config_dir(self, config_dir):
        """Test initialization with string path (converted to Path)"""
        loader = AgentConfigLoader(config_dir=str(config_dir))
        assert loader.config_dir == Path(config_dir)

    def test_init_sets_cache_to_empty_dict(self, config_dir):
        """Test that cache is initialized as empty dictionary"""
        loader = AgentConfigLoader(config_dir=config_dir)
        assert isinstance(loader._agents_cache, dict)
        assert len(loader._agents_cache) == 0

    def test_init_logs_config_dir(self, config_dir, caplog):
        """Test that initialization logs config directory"""
        with caplog.at_level(logging.INFO):
            loader = AgentConfigLoader(config_dir=config_dir)

        assert any("AgentConfigLoader initialized" in record.message
                   for record in caplog.records)


# ============================================================================
# LOAD_INDEX TESTS
# ============================================================================

class TestLoadIndex:
    """Test load_index() method"""

    def test_load_index_success(self, config_dir, valid_agent_index):
        """Test successfully loading valid index file"""
        index_path = config_dir / "agents_index.json"
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)
        result = loader.load_index()

        assert result == valid_agent_index
        assert result['total_agents'] == 2
        assert result['index_version'] == "2.0.0"
        assert len(result['agents']) == 2

    def test_load_index_caches_result(self, config_dir, valid_agent_index):
        """Test that index is cached after first load"""
        index_path = config_dir / "agents_index.json"
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)

        # First load
        result1 = loader.load_index()
        assert loader._index is not None

        # Second load should return cached value
        result2 = loader.load_index()
        assert result2 is result1
        assert id(result1) == id(result2)

    def test_load_index_file_not_found(self, config_dir):
        """Test loading index when file doesn't exist"""
        loader = AgentConfigLoader(config_dir=config_dir)

        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load_index()

        assert "Index file not found" in str(exc_info.value)
        assert "agents_index.json" in str(exc_info.value)

    def test_load_index_invalid_json(self, config_dir, invalid_json_data):
        """Test loading index with invalid JSON"""
        index_path = config_dir / "agents_index.json"
        index_path.write_text(invalid_json_data)

        loader = AgentConfigLoader(config_dir=config_dir)

        with pytest.raises(json.JSONDecodeError):
            loader.load_index()

    def test_load_index_empty_file(self, config_dir):
        """Test loading index from empty file"""
        index_path = config_dir / "agents_index.json"
        index_path.write_text("")

        loader = AgentConfigLoader(config_dir=config_dir)

        with pytest.raises(json.JSONDecodeError):
            loader.load_index()

    def test_load_index_logs_success(self, config_dir, valid_agent_index, caplog):
        """Test that successful index load is logged"""
        index_path = config_dir / "agents_index.json"
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)

        with caplog.at_level(logging.INFO):
            loader.load_index()

        assert any("Loaded agent index" in record.message
                   for record in caplog.records)
        assert any("2 agents" in record.message
                   for record in caplog.records)

    def test_load_index_logs_error_on_invalid_json(self, config_dir, invalid_json_data, caplog):
        """Test that JSON parsing errors are logged"""
        index_path = config_dir / "agents_index.json"
        index_path.write_text(invalid_json_data)

        loader = AgentConfigLoader(config_dir=config_dir)

        with caplog.at_level(logging.ERROR):
            with pytest.raises(json.JSONDecodeError):
                loader.load_index()

        assert any("Failed to parse index file" in record.message
                   for record in caplog.records)


# ============================================================================
# GET_AGENT_IDS TESTS
# ============================================================================

class TestGetAgentIds:
    """Test get_agent_ids() method"""

    def test_get_agent_ids_success(self, config_dir, valid_agent_index):
        """Test successfully retrieving agent IDs"""
        index_path = config_dir / "agents_index.json"
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)
        agent_ids = loader.get_agent_ids()

        assert len(agent_ids) == 2
        assert "test-agent-001" in agent_ids
        assert "test-agent-002" in agent_ids

    def test_get_agent_ids_empty_index(self, config_dir):
        """Test get_agent_ids with empty agents list"""
        empty_index = {
            "index_version": "2.0.0",
            "total_agents": 0,
            "agents": []
        }
        index_path = config_dir / "agents_index.json"
        index_path.write_text(json.dumps(empty_index))

        loader = AgentConfigLoader(config_dir=config_dir)
        agent_ids = loader.get_agent_ids()

        assert agent_ids == []

    def test_get_agent_ids_missing_agents_key(self, config_dir):
        """Test get_agent_ids when 'agents' key is missing"""
        incomplete_index = {
            "index_version": "2.0.0",
            "total_agents": 0
        }
        index_path = config_dir / "agents_index.json"
        index_path.write_text(json.dumps(incomplete_index))

        loader = AgentConfigLoader(config_dir=config_dir)
        agent_ids = loader.get_agent_ids()

        assert agent_ids == []


# ============================================================================
# GET_AGENT_BY_TYPE TESTS
# ============================================================================

class TestGetAgentByType:
    """Test get_agent_by_type() method"""

    def test_get_agent_by_type_code_quality(self, config_dir, valid_agent_index):
        """Test filtering agents by code_quality_security type"""
        index_path = config_dir / "agents_index.json"
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)
        agents = loader.get_agent_by_type(AgentType.CODE_QUALITY_SECURITY)

        assert len(agents) == 1
        assert "test-agent-001" in agents

    def test_get_agent_by_type_growth_marketing(self, config_dir, valid_agent_index):
        """Test filtering agents by growth_marketing_automation type"""
        index_path = config_dir / "agents_index.json"
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)
        agents = loader.get_agent_by_type(AgentType.GROWTH_MARKETING_AUTOMATION)

        assert len(agents) == 1
        assert "test-agent-002" in agents

    def test_get_agent_by_type_no_matches(self, config_dir, valid_agent_index):
        """Test filtering when no agents match the type"""
        index_path = config_dir / "agents_index.json"
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)
        agents = loader.get_agent_by_type(AgentType.VISUAL_CONTENT_GENERATION)

        assert agents == []

    def test_get_agent_by_type_all_types(self, config_dir):
        """Test filtering with agents of all types"""
        multi_type_index = {
            "index_version": "2.0.0",
            "total_agents": 4,
            "agents": [
                {"agent_id": "agent-1", "agent_type": "code_quality_security"},
                {"agent_id": "agent-2", "agent_type": "growth_marketing_automation"},
                {"agent_id": "agent-3", "agent_type": "data_analysis_intelligence"},
                {"agent_id": "agent-4", "agent_type": "visual_content_generation"}
            ]
        }
        index_path = config_dir / "agents_index.json"
        index_path.write_text(json.dumps(multi_type_index))

        loader = AgentConfigLoader(config_dir=config_dir)

        assert loader.get_agent_by_type(AgentType.CODE_QUALITY_SECURITY) == ["agent-1"]
        assert loader.get_agent_by_type(AgentType.GROWTH_MARKETING_AUTOMATION) == ["agent-2"]
        assert loader.get_agent_by_type(AgentType.DATA_ANALYSIS_INTELLIGENCE) == ["agent-3"]
        assert loader.get_agent_by_type(AgentType.VISUAL_CONTENT_GENERATION) == ["agent-4"]


# ============================================================================
# LOAD_AGENT TESTS
# ============================================================================

class TestLoadAgent:
    """Test load_agent() method"""

    def test_load_agent_success(self, config_dir, valid_agent_index, valid_agent_config, tmp_path):
        """Test successfully loading a valid agent configuration"""
        # Setup index
        index_path = config_dir / "agents_index.json"

        # Create agent config file
        agent_config_path = tmp_path / "test_agent.json"
        agent_config_path.write_text(json.dumps(valid_agent_config))

        # Update index with correct path
        valid_agent_index['agents'][0]['config_file'] = str(agent_config_path)
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)
        agent = loader.load_agent("test-agent-001")

        assert isinstance(agent, AgentConfiguration)
        assert agent.agent_id == "test-agent-001"
        assert agent.agent_name == "Test Agent"
        assert agent.agent_type == AgentType.CODE_QUALITY_SECURITY
        assert agent.status == AgentStatus.ACTIVE
        assert agent.version == "2.0.0"
        assert agent.maintainer == "DevSkyy Platform Team"

    def test_load_agent_caches_result(self, config_dir, valid_agent_index, valid_agent_config, tmp_path):
        """Test that loaded agent is cached"""
        # Setup
        index_path = config_dir / "agents_index.json"
        agent_config_path = tmp_path / "test_agent.json"
        agent_config_path.write_text(json.dumps(valid_agent_config))
        valid_agent_index['agents'][0]['config_file'] = str(agent_config_path)
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)

        # First load
        agent1 = loader.load_agent("test-agent-001")
        assert "test-agent-001" in loader._agents_cache

        # Second load should return cached instance
        agent2 = loader.load_agent("test-agent-001")
        assert agent2 is agent1
        assert id(agent1) == id(agent2)

    def test_load_agent_not_in_index(self, config_dir, valid_agent_index):
        """Test loading agent that doesn't exist in index"""
        index_path = config_dir / "agents_index.json"
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)

        with pytest.raises(ValueError) as exc_info:
            loader.load_agent("nonexistent-agent")

        assert "Agent ID not found in index" in str(exc_info.value)
        assert "nonexistent-agent" in str(exc_info.value)

    def test_load_agent_config_file_not_found(self, config_dir, valid_agent_index):
        """Test loading agent when config file doesn't exist"""
        index_path = config_dir / "agents_index.json"
        valid_agent_index['agents'][0]['config_file'] = "/nonexistent/path.json"
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)

        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load_agent("test-agent-001")

        assert "Agent config file not found" in str(exc_info.value)

    def test_load_agent_invalid_json(self, config_dir, valid_agent_index, tmp_path, invalid_json_data):
        """Test loading agent with invalid JSON config"""
        index_path = config_dir / "agents_index.json"
        agent_config_path = tmp_path / "test_agent.json"
        agent_config_path.write_text(invalid_json_data)
        valid_agent_index['agents'][0]['config_file'] = str(agent_config_path)
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)

        with pytest.raises(ValueError) as exc_info:
            loader.load_agent("test-agent-001")

        assert "Invalid agent configuration" in str(exc_info.value)

    def test_load_agent_missing_required_fields(self, config_dir, valid_agent_index,
                                                  tmp_path, agent_config_missing_required_fields):
        """Test loading agent with missing required fields"""
        index_path = config_dir / "agents_index.json"
        agent_config_path = tmp_path / "test_agent.json"
        agent_config_path.write_text(json.dumps(agent_config_missing_required_fields))
        valid_agent_index['agents'][0]['config_file'] = str(agent_config_path)
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)

        with pytest.raises(ValueError) as exc_info:
            loader.load_agent("test-agent-001")

        assert "Invalid agent configuration" in str(exc_info.value)

    def test_load_agent_parses_orchestration_commands(self, config_dir, valid_agent_index,
                                                       valid_agent_config, tmp_path):
        """Test that orchestration commands are properly parsed"""
        index_path = config_dir / "agents_index.json"
        agent_config_path = tmp_path / "test_agent.json"
        agent_config_path.write_text(json.dumps(valid_agent_config))
        valid_agent_index['agents'][0]['config_file'] = str(agent_config_path)
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)
        agent = loader.load_agent("test-agent-001")

        assert len(agent.orchestration_commands) == 2
        assert "PLAN" in agent.orchestration_commands
        assert "BUILD" in agent.orchestration_commands

        plan_cmd = agent.orchestration_commands["PLAN"]
        assert isinstance(plan_cmd, OrchestrationCommand)
        assert plan_cmd.command == "PLAN(scope)"
        assert plan_cmd.description == "Create job graph"
        assert plan_cmd.execution_mode == "synchronous"
        assert plan_cmd.required_inputs == ["scope"]
        assert plan_cmd.coverage_requirement == 90

    def test_load_agent_parses_performance_slos(self, config_dir, valid_agent_index,
                                                 valid_agent_config, tmp_path):
        """Test that performance SLOs are properly parsed"""
        index_path = config_dir / "agents_index.json"
        agent_config_path = tmp_path / "test_agent.json"
        agent_config_path.write_text(json.dumps(valid_agent_config))
        valid_agent_index['agents'][0]['config_file'] = str(agent_config_path)
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)
        agent = loader.load_agent("test-agent-001")

        slo = agent.performance_slos
        assert isinstance(slo, PerformanceSLO)
        assert slo.p95_latency_ms == 200
        assert slo.error_rate_percent == 0.5
        assert slo.test_coverage_percent == 90
        assert slo.secrets_in_repo == 0
        assert slo.page_load_time_ms == 1000
        assert slo.lighthouse_score == 95
        assert slo.conversion_rate_minimum == 0.02

    def test_load_agent_parses_optional_fields(self, config_dir, valid_agent_index,
                                                valid_agent_config, tmp_path):
        """Test that optional fields are properly parsed"""
        index_path = config_dir / "agents_index.json"
        agent_config_path = tmp_path / "test_agent.json"
        agent_config_path.write_text(json.dumps(valid_agent_config))
        valid_agent_index['agents'][0]['config_file'] = str(agent_config_path)
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)
        agent = loader.load_agent("test-agent-001")

        assert agent.wordpress_integration is not None
        assert agent.wordpress_integration['enabled'] is True
        assert agent.ci_cd_integration is not None
        assert agent.ci_cd_integration['platform'] == "GitHub Actions"

    def test_load_agent_minimal_config(self, config_dir, valid_agent_index,
                                        minimal_agent_config, tmp_path):
        """Test loading agent with minimal required fields"""
        index_path = config_dir / "agents_index.json"
        agent_config_path = tmp_path / "minimal_agent.json"
        agent_config_path.write_text(json.dumps(minimal_agent_config))

        # Add minimal agent to index
        valid_agent_index['agents'].append({
            "agent_id": "minimal-agent-001",
            "agent_name": "Minimal Agent",
            "agent_type": "data_analysis_intelligence",
            "config_file": str(agent_config_path),
            "status": "maintenance",
            "priority": 3
        })
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)
        agent = loader.load_agent("minimal-agent-001")

        assert agent.agent_id == "minimal-agent-001"
        assert agent.agent_type == AgentType.DATA_ANALYSIS_INTELLIGENCE
        assert agent.status == AgentStatus.MAINTENANCE
        assert agent.wordpress_integration is None
        assert agent.ab_testing_automation is None

    def test_load_agent_parses_datetime_fields(self, config_dir, valid_agent_index,
                                                valid_agent_config, tmp_path):
        """Test that datetime fields are properly parsed"""
        index_path = config_dir / "agents_index.json"
        agent_config_path = tmp_path / "test_agent.json"
        agent_config_path.write_text(json.dumps(valid_agent_config))
        valid_agent_index['agents'][0]['config_file'] = str(agent_config_path)
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)
        agent = loader.load_agent("test-agent-001")

        assert isinstance(agent.created_at, datetime)
        assert isinstance(agent.last_updated, datetime)
        assert agent.created_at.year == 2025
        assert agent.created_at.month == 11
        assert agent.created_at.day == 4

    def test_load_agent_logs_success(self, config_dir, valid_agent_index,
                                      valid_agent_config, tmp_path, caplog):
        """Test that successful agent load is logged"""
        index_path = config_dir / "agents_index.json"
        agent_config_path = tmp_path / "test_agent.json"
        agent_config_path.write_text(json.dumps(valid_agent_config))
        valid_agent_index['agents'][0]['config_file'] = str(agent_config_path)
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)

        with caplog.at_level(logging.INFO):
            loader.load_agent("test-agent-001")

        assert any("Loaded agent configuration" in record.message
                   and "test-agent-001" in record.message
                   for record in caplog.records)

    def test_load_agent_logs_cache_hit(self, config_dir, valid_agent_index,
                                        valid_agent_config, tmp_path, caplog):
        """Test that cache hits are logged at debug level"""
        index_path = config_dir / "agents_index.json"
        agent_config_path = tmp_path / "test_agent.json"
        agent_config_path.write_text(json.dumps(valid_agent_config))
        valid_agent_index['agents'][0]['config_file'] = str(agent_config_path)
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)

        # First load
        loader.load_agent("test-agent-001")

        # Second load with debug logging
        with caplog.at_level(logging.DEBUG):
            loader.load_agent("test-agent-001")

        assert any("Returning cached agent" in record.message
                   for record in caplog.records)

    def test_load_agent_invalid_agent_type(self, config_dir, valid_agent_index,
                                            valid_agent_config, tmp_path):
        """Test loading agent with invalid agent_type value"""
        index_path = config_dir / "agents_index.json"
        agent_config_path = tmp_path / "test_agent.json"

        # Set invalid agent type
        valid_agent_config['agent_type'] = "invalid_type"
        agent_config_path.write_text(json.dumps(valid_agent_config))
        valid_agent_index['agents'][0]['config_file'] = str(agent_config_path)
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)

        with pytest.raises(ValueError) as exc_info:
            loader.load_agent("test-agent-001")

        assert "Invalid agent configuration" in str(exc_info.value)

    def test_load_agent_invalid_status(self, config_dir, valid_agent_index,
                                        valid_agent_config, tmp_path):
        """Test loading agent with invalid status value"""
        index_path = config_dir / "agents_index.json"
        agent_config_path = tmp_path / "test_agent.json"

        # Set invalid status
        valid_agent_config['status'] = "invalid_status"
        agent_config_path.write_text(json.dumps(valid_agent_config))
        valid_agent_index['agents'][0]['config_file'] = str(agent_config_path)
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)

        with pytest.raises(ValueError) as exc_info:
            loader.load_agent("test-agent-001")

        assert "Invalid agent configuration" in str(exc_info.value)


# ============================================================================
# LOAD_ALL_AGENTS TESTS
# ============================================================================

class TestLoadAllAgents:
    """Test load_all_agents() method"""

    def test_load_all_agents_success(self, config_dir, valid_agent_index,
                                      valid_agent_config, minimal_agent_config, tmp_path):
        """Test successfully loading all agents"""
        index_path = config_dir / "agents_index.json"

        # Create agent config files
        agent1_path = tmp_path / "agent1.json"
        agent1_path.write_text(json.dumps(valid_agent_config))

        agent2_config = minimal_agent_config.copy()
        agent2_config['agent_id'] = "test-agent-002"
        agent2_config['agent_type'] = "growth_marketing_automation"
        agent2_path = tmp_path / "agent2.json"
        agent2_path.write_text(json.dumps(agent2_config))

        # Update index
        valid_agent_index['agents'][0]['config_file'] = str(agent1_path)
        valid_agent_index['agents'][1]['config_file'] = str(agent2_path)
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)
        agents = loader.load_all_agents()

        assert len(agents) == 2
        assert "test-agent-001" in agents
        assert "test-agent-002" in agents
        assert isinstance(agents["test-agent-001"], AgentConfiguration)
        assert isinstance(agents["test-agent-002"], AgentConfiguration)

    def test_load_all_agents_empty_index(self, config_dir):
        """Test load_all_agents with empty index"""
        empty_index = {
            "index_version": "2.0.0",
            "total_agents": 0,
            "agents": []
        }
        index_path = config_dir / "agents_index.json"
        index_path.write_text(json.dumps(empty_index))

        loader = AgentConfigLoader(config_dir=config_dir)
        agents = loader.load_all_agents()

        assert agents == {}

    def test_load_all_agents_skips_invalid_agents(self, config_dir, valid_agent_index,
                                                   valid_agent_config, tmp_path, caplog):
        """Test that load_all_agents continues when some agents fail to load"""
        index_path = config_dir / "agents_index.json"

        # Create one valid agent
        agent1_path = tmp_path / "agent1.json"
        agent1_path.write_text(json.dumps(valid_agent_config))

        # Second agent has invalid path (will fail)
        valid_agent_index['agents'][0]['config_file'] = str(agent1_path)
        valid_agent_index['agents'][1]['config_file'] = "/nonexistent/path.json"
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)

        with caplog.at_level(logging.WARNING):
            agents = loader.load_all_agents()

        # Should load only the valid agent
        assert len(agents) == 1
        assert "test-agent-001" in agents
        assert "test-agent-002" not in agents

        # Should log warning about failed agent
        assert any("Failed to load agent" in record.message
                   and "test-agent-002" in record.message
                   for record in caplog.records)

    def test_load_all_agents_logs_summary(self, config_dir, valid_agent_index,
                                           valid_agent_config, tmp_path, caplog):
        """Test that load_all_agents logs a summary"""
        index_path = config_dir / "agents_index.json"
        agent1_path = tmp_path / "agent1.json"
        agent1_path.write_text(json.dumps(valid_agent_config))
        valid_agent_index['agents'][0]['config_file'] = str(agent1_path)
        valid_agent_index['agents'][1]['config_file'] = "/nonexistent/path.json"
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)

        with caplog.at_level(logging.INFO):
            loader.load_all_agents()

        assert any("Loaded 1/2 agents" in record.message
                   for record in caplog.records)


# ============================================================================
# VALIDATE_AGENT TESTS
# ============================================================================

class TestValidateAgent:
    """Test validate_agent() method - Truth Protocol compliance"""

    def test_validate_agent_success(self, config_dir, valid_agent_index,
                                     valid_agent_config, tmp_path):
        """Test validating a fully compliant agent"""
        index_path = config_dir / "agents_index.json"
        agent_config_path = tmp_path / "test_agent.json"
        agent_config_path.write_text(json.dumps(valid_agent_config))
        valid_agent_index['agents'][0]['config_file'] = str(agent_config_path)
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)
        validation = loader.validate_agent("test-agent-001")

        assert validation['valid'] is True
        assert validation['agent_id'] == "test-agent-001"
        assert len(validation['errors']) == 0

    def test_validate_agent_missing_truth_protocol_keys(self, config_dir, valid_agent_index,
                                                         valid_agent_config, tmp_path):
        """Test validation fails when Truth Protocol keys are missing"""
        index_path = config_dir / "agents_index.json"
        agent_config_path = tmp_path / "test_agent.json"

        # Remove required Truth Protocol keys
        valid_agent_config['truth_protocol_compliance'] = {
            "never_guess": True,
            "pin_versions": True
            # Missing: cite_standards, no_hardcoded_secrets, etc.
        }
        agent_config_path.write_text(json.dumps(valid_agent_config))
        valid_agent_index['agents'][0]['config_file'] = str(agent_config_path)
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)
        validation = loader.validate_agent("test-agent-001")

        assert validation['valid'] is False
        assert len(validation['errors']) > 0
        assert any("Missing Truth Protocol key" in error for error in validation['errors'])

    def test_validate_agent_low_test_coverage(self, config_dir, valid_agent_index,
                                               valid_agent_config, tmp_path):
        """Test validation warns on test coverage < 90%"""
        index_path = config_dir / "agents_index.json"
        agent_config_path = tmp_path / "test_agent.json"

        # Set low test coverage
        valid_agent_config['truth_protocol_compliance']['test_coverage_minimum'] = 75
        agent_config_path.write_text(json.dumps(valid_agent_config))
        valid_agent_index['agents'][0]['config_file'] = str(agent_config_path)
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)
        validation = loader.validate_agent("test-agent-001")

        assert validation['valid'] is True  # Warning, not error
        assert len(validation['warnings']) > 0
        assert any("Test coverage minimum" in warning and "< 90%" in warning
                   for warning in validation['warnings'])

    def test_validate_agent_high_latency(self, config_dir, valid_agent_index,
                                          valid_agent_config, tmp_path):
        """Test validation warns on P95 latency > 500ms"""
        index_path = config_dir / "agents_index.json"
        agent_config_path = tmp_path / "test_agent.json"

        # Set high latency
        valid_agent_config['performance_slos']['p95_latency_ms'] = 600
        agent_config_path.write_text(json.dumps(valid_agent_config))
        valid_agent_index['agents'][0]['config_file'] = str(agent_config_path)
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)
        validation = loader.validate_agent("test-agent-001")

        assert validation['valid'] is True  # Warning, not error
        assert len(validation['warnings']) > 0
        assert any("P95 latency" in warning and "> 500ms" in warning
                   for warning in validation['warnings'])

    def test_validate_agent_secrets_in_repo(self, config_dir, valid_agent_index,
                                             valid_agent_config, tmp_path):
        """Test validation fails when secrets_in_repo > 0"""
        index_path = config_dir / "agents_index.json"
        agent_config_path = tmp_path / "test_agent.json"

        # Set secrets in repo
        valid_agent_config['performance_slos']['secrets_in_repo'] = 3
        agent_config_path.write_text(json.dumps(valid_agent_config))
        valid_agent_index['agents'][0]['config_file'] = str(agent_config_path)
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)
        validation = loader.validate_agent("test-agent-001")

        assert validation['valid'] is False
        assert len(validation['errors']) > 0
        assert any("Secrets in repo" in error and "must be 0" in error
                   for error in validation['errors'])

    def test_validate_agent_no_orchestration_commands(self, config_dir, valid_agent_index,
                                                       valid_agent_config, tmp_path):
        """Test validation warns when no orchestration commands defined"""
        index_path = config_dir / "agents_index.json"
        agent_config_path = tmp_path / "test_agent.json"

        # Remove orchestration commands
        valid_agent_config['orchestration_commands'] = {}
        agent_config_path.write_text(json.dumps(valid_agent_config))
        valid_agent_index['agents'][0]['config_file'] = str(agent_config_path)
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)
        validation = loader.validate_agent("test-agent-001")

        assert validation['valid'] is True  # Warning, not error
        assert len(validation['warnings']) > 0
        assert any("No orchestration commands" in warning
                   for warning in validation['warnings'])

    def test_validate_agent_multiple_errors_and_warnings(self, config_dir, valid_agent_index,
                                                          valid_agent_config, tmp_path):
        """Test validation with multiple errors and warnings"""
        index_path = config_dir / "agents_index.json"
        agent_config_path = tmp_path / "test_agent.json"

        # Create multiple validation issues
        valid_agent_config['truth_protocol_compliance'] = {
            "never_guess": True
            # Missing most required keys
        }
        valid_agent_config['performance_slos']['secrets_in_repo'] = 5
        valid_agent_config['performance_slos']['p95_latency_ms'] = 1000
        valid_agent_config['orchestration_commands'] = {}

        agent_config_path.write_text(json.dumps(valid_agent_config))
        valid_agent_index['agents'][0]['config_file'] = str(agent_config_path)
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)
        validation = loader.validate_agent("test-agent-001")

        assert validation['valid'] is False
        assert len(validation['errors']) > 1
        assert len(validation['warnings']) > 1

    def test_validate_agent_logs_result(self, config_dir, valid_agent_index,
                                         valid_agent_config, tmp_path, caplog):
        """Test that validation result is logged"""
        index_path = config_dir / "agents_index.json"
        agent_config_path = tmp_path / "test_agent.json"
        agent_config_path.write_text(json.dumps(valid_agent_config))
        valid_agent_index['agents'][0]['config_file'] = str(agent_config_path)
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)

        with caplog.at_level(logging.INFO):
            loader.validate_agent("test-agent-001")

        assert any("Validation for test-agent-001" in record.message
                   and "PASS" in record.message
                   for record in caplog.records)

    @pytest.mark.parametrize("missing_key", [
        "never_guess",
        "pin_versions",
        "cite_standards",
        "no_hardcoded_secrets",
        "rbac_enforcement",
        "test_coverage_minimum",
        "no_skip_rule",
        "error_ledger_required"
    ])
    def test_validate_agent_each_required_key(self, config_dir, valid_agent_index,
                                               valid_agent_config, tmp_path, missing_key):
        """Test validation fails for each individual missing Truth Protocol key"""
        index_path = config_dir / "agents_index.json"
        agent_config_path = tmp_path / "test_agent.json"

        # Remove specific key
        tp_compliance = valid_agent_config['truth_protocol_compliance'].copy()
        del tp_compliance[missing_key]
        valid_agent_config['truth_protocol_compliance'] = tp_compliance

        agent_config_path.write_text(json.dumps(valid_agent_config))
        valid_agent_index['agents'][0]['config_file'] = str(agent_config_path)
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)
        validation = loader.validate_agent("test-agent-001")

        assert validation['valid'] is False
        assert any(missing_key in error for error in validation['errors'])


# ============================================================================
# GET_QUICK_REFERENCE TESTS
# ============================================================================

class TestGetQuickReference:
    """Test get_quick_reference() method"""

    def test_get_quick_reference_success(self, config_dir, valid_agent_index):
        """Test successfully retrieving quick reference"""
        index_path = config_dir / "agents_index.json"
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)
        quick_ref = loader.get_quick_reference()

        assert isinstance(quick_ref, dict)
        assert quick_ref['code_quality'] == "test-agent-001"
        assert quick_ref['marketing_growth'] == "test-agent-002"

    def test_get_quick_reference_empty(self, config_dir):
        """Test get_quick_reference when quick_reference is empty"""
        index = {
            "index_version": "2.0.0",
            "total_agents": 0,
            "agents": [],
            "quick_reference": {}
        }
        index_path = config_dir / "agents_index.json"
        index_path.write_text(json.dumps(index))

        loader = AgentConfigLoader(config_dir=config_dir)
        quick_ref = loader.get_quick_reference()

        assert quick_ref == {}

    def test_get_quick_reference_missing_key(self, config_dir):
        """Test get_quick_reference when quick_reference key is missing"""
        index = {
            "index_version": "2.0.0",
            "total_agents": 0,
            "agents": []
        }
        index_path = config_dir / "agents_index.json"
        index_path.write_text(json.dumps(index))

        loader = AgentConfigLoader(config_dir=config_dir)
        quick_ref = loader.get_quick_reference()

        assert quick_ref == {}


# ============================================================================
# CACHE BEHAVIOR TESTS
# ============================================================================

class TestCacheBehavior:
    """Test cache management functionality"""

    def test_clear_cache_clears_agents(self, config_dir, valid_agent_index,
                                        valid_agent_config, tmp_path):
        """Test that clear_cache() clears the agents cache"""
        index_path = config_dir / "agents_index.json"
        agent_config_path = tmp_path / "test_agent.json"
        agent_config_path.write_text(json.dumps(valid_agent_config))
        valid_agent_index['agents'][0]['config_file'] = str(agent_config_path)
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)

        # Load an agent to populate cache
        loader.load_agent("test-agent-001")
        assert len(loader._agents_cache) == 1

        # Clear cache
        loader.clear_cache()
        assert len(loader._agents_cache) == 0

    def test_clear_cache_clears_index(self, config_dir, valid_agent_index):
        """Test that clear_cache() clears the index cache"""
        index_path = config_dir / "agents_index.json"
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)

        # Load index to populate cache
        loader.load_index()
        assert loader._index is not None

        # Clear cache
        loader.clear_cache()
        assert loader._index is None

    def test_clear_cache_logs_message(self, config_dir, caplog):
        """Test that clear_cache() logs a message"""
        loader = AgentConfigLoader(config_dir=config_dir)

        with caplog.at_level(logging.INFO):
            loader.clear_cache()

        assert any("Agent configuration cache cleared" in record.message
                   for record in caplog.records)

    def test_cache_isolation_between_instances(self, config_dir, valid_agent_index,
                                                valid_agent_config, tmp_path):
        """Test that cache is isolated between different loader instances"""
        index_path = config_dir / "agents_index.json"
        agent_config_path = tmp_path / "test_agent.json"
        agent_config_path.write_text(json.dumps(valid_agent_config))
        valid_agent_index['agents'][0]['config_file'] = str(agent_config_path)
        index_path.write_text(json.dumps(valid_agent_index))

        loader1 = AgentConfigLoader(config_dir=config_dir)
        loader2 = AgentConfigLoader(config_dir=config_dir)

        # Load agent in loader1
        loader1.load_agent("test-agent-001")
        assert len(loader1._agents_cache) == 1
        assert len(loader2._agents_cache) == 0

        # Load agent in loader2
        loader2.load_agent("test-agent-001")
        assert len(loader2._agents_cache) == 1

        # Clear loader1 cache
        loader1.clear_cache()
        assert len(loader1._agents_cache) == 0
        assert len(loader2._agents_cache) == 1


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test comprehensive error handling scenarios"""

    def test_load_agent_with_empty_agent_id(self, config_dir, valid_agent_index):
        """Test loading agent with empty agent_id"""
        index_path = config_dir / "agents_index.json"
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)

        with pytest.raises(ValueError) as exc_info:
            loader.load_agent("")

        assert "Agent ID not found in index" in str(exc_info.value)

    def test_load_agent_with_none_agent_id(self, config_dir, valid_agent_index):
        """Test loading agent with None agent_id"""
        index_path = config_dir / "agents_index.json"
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)

        with pytest.raises((ValueError, AttributeError)):
            loader.load_agent(None)

    def test_malformed_datetime_in_config(self, config_dir, valid_agent_index,
                                           valid_agent_config, tmp_path):
        """Test loading agent with malformed datetime fields"""
        index_path = config_dir / "agents_index.json"
        agent_config_path = tmp_path / "test_agent.json"

        # Set invalid datetime
        valid_agent_config['created_at'] = "not-a-datetime"
        agent_config_path.write_text(json.dumps(valid_agent_config))
        valid_agent_index['agents'][0]['config_file'] = str(agent_config_path)
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)

        with pytest.raises(ValueError) as exc_info:
            loader.load_agent("test-agent-001")

        assert "Invalid agent configuration" in str(exc_info.value)

    def test_file_permission_error(self, config_dir, valid_agent_index, tmp_path):
        """Test handling of file permission errors"""
        index_path = config_dir / "agents_index.json"
        agent_config_path = tmp_path / "test_agent.json"

        # Create file but make it unreadable (Unix-like systems only)
        agent_config_path.write_text("{}")
        try:
            agent_config_path.chmod(0o000)

            valid_agent_index['agents'][0]['config_file'] = str(agent_config_path)
            index_path.write_text(json.dumps(valid_agent_index))

            loader = AgentConfigLoader(config_dir=config_dir)

            with pytest.raises((ValueError, PermissionError)):
                loader.load_agent("test-agent-001")
        finally:
            # Restore permissions for cleanup
            agent_config_path.chmod(0o644)

    def test_unicode_handling_in_config(self, config_dir, valid_agent_index,
                                         valid_agent_config, tmp_path):
        """Test handling of unicode characters in configuration"""
        index_path = config_dir / "agents_index.json"
        agent_config_path = tmp_path / "test_agent.json"

        # Add unicode characters
        valid_agent_config['agent_name'] = "Test Agent 测试代理 🤖"
        valid_agent_config['maintainer'] = "DevSkyy 平台团队"

        agent_config_path.write_text(json.dumps(valid_agent_config), encoding='utf-8')
        valid_agent_index['agents'][0]['config_file'] = str(agent_config_path)
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)
        agent = loader.load_agent("test-agent-001")

        assert "测试代理" in agent.agent_name
        assert "平台团队" in agent.maintainer


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple methods"""

    def test_full_workflow_load_validate_cache(self, config_dir, valid_agent_index,
                                                valid_agent_config, tmp_path):
        """Test complete workflow: load index, load agent, validate, use cache"""
        index_path = config_dir / "agents_index.json"
        agent_config_path = tmp_path / "test_agent.json"
        agent_config_path.write_text(json.dumps(valid_agent_config))
        valid_agent_index['agents'][0]['config_file'] = str(agent_config_path)
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)

        # Load index
        index = loader.load_index()
        assert index['total_agents'] == 2

        # Get agent IDs
        agent_ids = loader.get_agent_ids()
        assert len(agent_ids) == 2

        # Load specific agent
        agent = loader.load_agent("test-agent-001")
        assert agent.agent_id == "test-agent-001"

        # Validate agent
        validation = loader.validate_agent("test-agent-001")
        assert validation['valid'] is True

        # Load from cache
        agent_cached = loader.load_agent("test-agent-001")
        assert agent_cached is agent

        # Get quick reference
        quick_ref = loader.get_quick_reference()
        assert quick_ref['code_quality'] == "test-agent-001"

        # Clear cache and reload
        loader.clear_cache()
        agent_fresh = loader.load_agent("test-agent-001")
        assert agent_fresh is not agent

    def test_filter_by_type_and_load(self, config_dir, valid_agent_index,
                                      valid_agent_config, minimal_agent_config, tmp_path):
        """Test filtering agents by type and loading them"""
        index_path = config_dir / "agents_index.json"

        # Setup two different agent types
        agent1_path = tmp_path / "agent1.json"
        agent1_path.write_text(json.dumps(valid_agent_config))

        agent2_config = minimal_agent_config.copy()
        agent2_config['agent_id'] = "test-agent-002"
        agent2_config['agent_type'] = "growth_marketing_automation"
        agent2_path = tmp_path / "agent2.json"
        agent2_path.write_text(json.dumps(agent2_config))

        valid_agent_index['agents'][0]['config_file'] = str(agent1_path)
        valid_agent_index['agents'][1]['config_file'] = str(agent2_path)
        index_path.write_text(json.dumps(valid_agent_index))

        loader = AgentConfigLoader(config_dir=config_dir)

        # Get code quality agents
        code_quality_ids = loader.get_agent_by_type(AgentType.CODE_QUALITY_SECURITY)
        assert len(code_quality_ids) == 1

        # Load and validate
        agent = loader.load_agent(code_quality_ids[0])
        assert agent.agent_type == AgentType.CODE_QUALITY_SECURITY

        validation = loader.validate_agent(code_quality_ids[0])
        assert validation['valid'] is True


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================

class TestParametrized:
    """Parametrized tests for comprehensive coverage"""

    @pytest.mark.parametrize("agent_type,expected_value", [
        (AgentType.CODE_QUALITY_SECURITY, "code_quality_security"),
        (AgentType.GROWTH_MARKETING_AUTOMATION, "growth_marketing_automation"),
        (AgentType.DATA_ANALYSIS_INTELLIGENCE, "data_analysis_intelligence"),
        (AgentType.VISUAL_CONTENT_GENERATION, "visual_content_generation"),
    ])
    def test_agent_type_enum_values(self, agent_type, expected_value):
        """Test all AgentType enum values"""
        assert agent_type.value == expected_value

    @pytest.mark.parametrize("agent_status,expected_value", [
        (AgentStatus.ACTIVE, "active"),
        (AgentStatus.INACTIVE, "inactive"),
        (AgentStatus.MAINTENANCE, "maintenance"),
        (AgentStatus.DEGRADED, "degraded"),
    ])
    def test_agent_status_enum_values(self, agent_status, expected_value):
        """Test all AgentStatus enum values"""
        assert agent_status.value == expected_value

    @pytest.mark.parametrize("slo_field,value", [
        ("p95_latency_ms", 200),
        ("error_rate_percent", 0.5),
        ("test_coverage_percent", 90),
        ("secrets_in_repo", 0),
        ("page_load_time_ms", 1000),
        ("lighthouse_score", 95),
        ("conversion_rate_minimum", 0.02),
    ])
    def test_performance_slo_fields(self, slo_field, value):
        """Test all PerformanceSLO fields"""
        slo = PerformanceSLO(**{slo_field: value})
        assert getattr(slo, slo_field) == value


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=agents.loader", "--cov-report=term-missing"])
