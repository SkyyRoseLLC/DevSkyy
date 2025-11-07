#!/usr/bin/env python3
"""
Pytest fixtures for agent routing system tests

Truth Protocol Compliance: CLAUDE.md
"""

import json
import pytest
import tempfile
from pathlib import Path
from typing import Dict, List
import sys

# Add project root to path BEFORE any other imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agents.loader import AgentConfigLoader, AgentConfig
from agents.router import AgentRouter, TaskRequest, TaskType


@pytest.fixture
def temp_config_dir(tmp_path: Path) -> Path:
    """
    Create temporary config directory with sample agent configs

    Args:
        tmp_path: Pytest temporary path fixture

    Returns:
        Path to temporary config directory
    """
    config_dir = tmp_path / "config" / "agents"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


@pytest.fixture
def sample_scanner_config() -> Dict:
    """Sample scanner agent configuration"""
    return {
        "agent_id": "test_scanner",
        "agent_type": "security",
        "name": "Test Scanner",
        "capabilities": ["security_scan", "vulnerability_scan", "code_review"],
        "priority": 80,
        "available": True,
        "description": "Test scanner agent for testing",
        "metadata": {"version": "1.0.0", "test": True}
    }


@pytest.fixture
def sample_fixer_config() -> Dict:
    """Sample fixer agent configuration"""
    return {
        "agent_id": "test_fixer",
        "agent_type": "code_repair",
        "name": "Test Fixer",
        "capabilities": ["code_fix", "auto_fix", "code_refactor"],
        "priority": 75,
        "available": True,
        "description": "Test fixer agent for testing",
        "metadata": {"version": "1.0.0", "test": True}
    }


@pytest.fixture
def sample_ml_config() -> Dict:
    """Sample ML agent configuration"""
    return {
        "agent_id": "test_ml",
        "agent_type": "machine_learning",
        "name": "Test ML System",
        "capabilities": ["ml_training", "ml_inference", "pattern_recognition"],
        "priority": 85,
        "available": True,
        "description": "Test ML agent for testing",
        "metadata": {"version": "1.0.0", "test": True}
    }


@pytest.fixture
def sample_unavailable_config() -> Dict:
    """Sample unavailable agent configuration"""
    return {
        "agent_id": "test_unavailable",
        "agent_type": "general",
        "name": "Test Unavailable",
        "capabilities": ["general"],
        "priority": 50,
        "available": False,
        "description": "Test unavailable agent",
        "metadata": {"version": "1.0.0", "test": True}
    }


@pytest.fixture
def invalid_config_missing_field() -> Dict:
    """Invalid config missing required field"""
    return {
        "agent_id": "invalid",
        "agent_type": "test",
        "name": "Invalid Agent"
        # Missing: capabilities, priority
    }


@pytest.fixture
def invalid_config_bad_priority() -> Dict:
    """Invalid config with out-of-range priority"""
    return {
        "agent_id": "invalid_priority",
        "agent_type": "test",
        "name": "Invalid Priority",
        "capabilities": ["test"],
        "priority": 150,  # Out of range (1-100)
        "available": True
    }


@pytest.fixture
def invalid_config_empty_capabilities() -> Dict:
    """Invalid config with empty capabilities"""
    return {
        "agent_id": "invalid_caps",
        "agent_type": "test",
        "name": "Invalid Capabilities",
        "capabilities": [],  # Empty list not allowed
        "priority": 50,
        "available": True
    }


@pytest.fixture
def config_loader(temp_config_dir: Path, sample_scanner_config: Dict,
                  sample_fixer_config: Dict, sample_ml_config: Dict) -> AgentConfigLoader:
    """
    Create AgentConfigLoader with sample configs

    Args:
        temp_config_dir: Temporary config directory
        sample_scanner_config: Scanner config
        sample_fixer_config: Fixer config
        sample_ml_config: ML config

    Returns:
        Configured AgentConfigLoader instance
    """
    # Write sample configs to temp directory
    configs = [
        ("test_scanner.json", sample_scanner_config),
        ("test_fixer.json", sample_fixer_config),
        ("test_ml.json", sample_ml_config),
    ]

    for filename, config in configs:
        config_path = temp_config_dir / filename
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

    return AgentConfigLoader(config_dir=temp_config_dir)


@pytest.fixture
def router(config_loader: AgentConfigLoader) -> AgentRouter:
    """
    Create AgentRouter with sample agents

    Args:
        config_loader: Configured AgentConfigLoader

    Returns:
        Configured AgentRouter instance
    """
    return AgentRouter(config_loader=config_loader)


@pytest.fixture
def sample_task() -> TaskRequest:
    """Sample task request for testing"""
    return TaskRequest(
        task_type=TaskType.SECURITY_SCAN,
        description="Scan Python code for security vulnerabilities",
        priority=80,
        metadata={"test": True}
    )


@pytest.fixture
def batch_tasks() -> List[TaskRequest]:
    """Batch of task requests for testing"""
    return [
        TaskRequest(
            task_type=TaskType.SECURITY_SCAN,
            description="Security scan task",
            priority=80
        ),
        TaskRequest(
            task_type=TaskType.CODE_FIX,
            description="Fix code issues",
            priority=75
        ),
        TaskRequest(
            task_type=TaskType.ML_TRAINING,
            description="Train ML model",
            priority=85
        ),
        TaskRequest(
            task_type=TaskType.CODE_GENERATION,
            description="Generate code",
            priority=60
        ),
        TaskRequest(
            task_type=TaskType.GENERAL_TASK,
            description="General task",
            priority=50
        )
    ]


@pytest.fixture
def valid_agent_config() -> AgentConfig:
    """Valid AgentConfig instance for testing"""
    return AgentConfig(
        agent_id="valid_agent",
        agent_type="test",
        name="Valid Agent",
        capabilities=["test_capability", "another_capability"],
        priority=70,
        available=True,
        description="Valid test agent",
        metadata={"version": "1.0.0"}
    )


@pytest.fixture
def high_priority_task() -> TaskRequest:
    """High priority task request"""
    return TaskRequest(
        task_type=TaskType.SECURITY_SCAN,
        description="Critical security scan",
        priority=95,
        metadata={"urgent": True}
    )


@pytest.fixture
def low_priority_task() -> TaskRequest:
    """Low priority task request"""
    return TaskRequest(
        task_type=TaskType.DOCUMENTATION_GENERATION,
        description="Generate documentation",
        priority=30,
        metadata={"optional": True}
    )
