#!/usr/bin/env python3
"""
Agent Configuration Loader
Programmatically loads and validates agent configurations from JSON files

References: /Users/coreyfoster/Desktop/CLAUDE_20-10_MASTER.md
Truth Protocol Compliance: Rules 1, 2, 3, 5, 10
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Add parent directory to path for standalone execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Agent type enumeration"""
    CODE_QUALITY_SECURITY = "code_quality_security"
    GROWTH_MARKETING_AUTOMATION = "growth_marketing_automation"
    DATA_ANALYSIS_INTELLIGENCE = "data_analysis_intelligence"
    VISUAL_CONTENT_GENERATION = "visual_content_generation"


class AgentStatus(Enum):
    """Agent status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"


@dataclass
class OrchestrationCommand:
    """Orchestration command configuration"""
    command: str
    description: str
    execution_mode: str
    required_inputs: List[str] = field(default_factory=list)
    validation_steps: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    coverage_requirement: Optional[int] = None


@dataclass
class PerformanceSLO:
    """Performance SLO configuration"""
    p95_latency_ms: Optional[int] = None
    error_rate_percent: Optional[float] = None
    test_coverage_percent: Optional[int] = None
    secrets_in_repo: int = 0
    page_load_time_ms: Optional[int] = None
    lighthouse_score: Optional[int] = None
    conversion_rate_minimum: Optional[float] = None


@dataclass
class AgentConfiguration:
    """Complete agent configuration"""
    agent_id: str
    agent_name: str
    agent_type: AgentType
    version: str
    master_document: str
    status: AgentStatus
    composition: Dict[str, str]
    capabilities: Dict[str, Any]
    truth_protocol_compliance: Dict[str, Any]
    orchestration_commands: Dict[str, OrchestrationCommand]
    performance_slos: PerformanceSLO
    monitoring: Dict[str, Any]
    deliverables_per_cycle: List[str]
    created_at: datetime
    last_updated: datetime
    maintainer: str

    # Optional fields
    wordpress_integration: Optional[Dict[str, Any]] = None
    ab_testing_automation: Optional[Dict[str, Any]] = None
    data_retrieval: Optional[Dict[str, Any]] = None
    model_management: Optional[Dict[str, Any]] = None
    ci_cd_integration: Optional[Dict[str, Any]] = None


class AgentConfigLoader:
    """
    Loads and validates agent configurations from JSON files

    Usage:
        loader = AgentConfigLoader()
        agent = loader.load_agent("professors-of-code-001")
        all_agents = loader.load_all_agents()
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize agent configuration loader

        Args:
            config_dir: Directory containing agent config files
                       Defaults to agents/config/ relative to this file
        """
        if config_dir is None:
            self.config_dir = Path(__file__).parent / "config"
        else:
            self.config_dir = Path(config_dir)

        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")

        self.index_path = self.config_dir / "agents_index.json"
        self._index = None
        self._agents_cache: Dict[str, AgentConfiguration] = {}

        logger.info(f"AgentConfigLoader initialized with config_dir: {self.config_dir}")

    def load_index(self) -> Dict[str, Any]:
        """
        Load the agent index file

        Returns:
            Dict containing agent index metadata

        Raises:
            FileNotFoundError: If index file doesn't exist
            json.JSONDecodeError: If index file is invalid JSON
        """
        if self._index is not None:
            return self._index

        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")

        try:
            with open(self.index_path, 'r') as f:
                self._index = json.load(f)
            logger.info(f"Loaded agent index: {self._index.get('total_agents', 0)} agents")
            return self._index
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse index file: {e}")
            raise

    def get_agent_ids(self) -> List[str]:
        """
        Get list of all agent IDs

        Returns:
            List of agent IDs
        """
        index = self.load_index()
        return [agent['agent_id'] for agent in index.get('agents', [])]

    def get_agent_by_type(self, agent_type: AgentType) -> List[str]:
        """
        Get agent IDs by type

        Args:
            agent_type: Type of agents to retrieve

        Returns:
            List of agent IDs matching the type
        """
        index = self.load_index()
        return [
            agent['agent_id']
            for agent in index.get('agents', [])
            if agent.get('agent_type') == agent_type.value
        ]

    def load_agent(self, agent_id: str) -> AgentConfiguration:
        """
        Load a specific agent configuration

        Args:
            agent_id: Agent ID to load (e.g., "professors-of-code-001")

        Returns:
            AgentConfiguration object

        Raises:
            FileNotFoundError: If agent config file doesn't exist
            ValueError: If agent configuration is invalid
        """
        # Check cache first
        if agent_id in self._agents_cache:
            logger.debug(f"Returning cached agent: {agent_id}")
            return self._agents_cache[agent_id]

        # Find config file path from index
        index = self.load_index()
        agent_meta = None
        for agent in index.get('agents', []):
            if agent['agent_id'] == agent_id:
                agent_meta = agent
                break

        if agent_meta is None:
            raise ValueError(f"Agent ID not found in index: {agent_id}")

        config_file = Path(agent_meta['config_file'])

        if not config_file.exists():
            raise FileNotFoundError(f"Agent config file not found: {config_file}")

        # Load and parse configuration
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)

            # Parse orchestration commands
            orchestration_commands = {}
            for cmd_name, cmd_data in config_data.get('orchestration_commands', {}).items():
                orchestration_commands[cmd_name] = OrchestrationCommand(
                    command=cmd_data.get('command', ''),
                    description=cmd_data.get('description', ''),
                    execution_mode=cmd_data.get('execution_mode', 'synchronous'),
                    required_inputs=cmd_data.get('required_inputs', []),
                    validation_steps=cmd_data.get('validation_steps', []),
                    tools=cmd_data.get('tools', []),
                    coverage_requirement=cmd_data.get('coverage_requirement')
                )

            # Parse performance SLOs
            slo_data = config_data.get('performance_slos', {})
            performance_slos = PerformanceSLO(
                p95_latency_ms=slo_data.get('p95_latency_ms'),
                error_rate_percent=slo_data.get('error_rate_percent'),
                test_coverage_percent=slo_data.get('test_coverage_percent'),
                secrets_in_repo=slo_data.get('secrets_in_repo', 0),
                page_load_time_ms=slo_data.get('page_load_time_ms'),
                lighthouse_score=slo_data.get('lighthouse_score'),
                conversion_rate_minimum=slo_data.get('conversion_rate_minimum')
            )

            # Create AgentConfiguration
            agent_config = AgentConfiguration(
                agent_id=config_data['agent_id'],
                agent_name=config_data['agent_name'],
                agent_type=AgentType(config_data['agent_type']),
                version=config_data['version'],
                master_document=config_data['master_document'],
                status=AgentStatus(config_data['status']),
                composition=config_data['composition'],
                capabilities=config_data['capabilities'],
                truth_protocol_compliance=config_data['truth_protocol_compliance'],
                orchestration_commands=orchestration_commands,
                performance_slos=performance_slos,
                monitoring=config_data.get('monitoring', {}),
                deliverables_per_cycle=config_data.get('deliverables_per_cycle', []),
                created_at=datetime.fromisoformat(config_data['created_at'].replace('Z', '+00:00')),
                last_updated=datetime.fromisoformat(config_data['last_updated'].replace('Z', '+00:00')),
                maintainer=config_data['maintainer'],
                wordpress_integration=config_data.get('wordpress_integration'),
                ab_testing_automation=config_data.get('ab_testing_automation'),
                data_retrieval=config_data.get('data_retrieval'),
                model_management=config_data.get('model_management'),
                ci_cd_integration=config_data.get('ci_cd_integration')
            )

            # Cache the configuration
            self._agents_cache[agent_id] = agent_config
            logger.info(f"Loaded agent configuration: {agent_id}")

            return agent_config

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to load agent {agent_id}: {e}")
            raise ValueError(f"Invalid agent configuration for {agent_id}: {e}")

    def load_all_agents(self) -> Dict[str, AgentConfiguration]:
        """
        Load all agent configurations

        Returns:
            Dict mapping agent_id to AgentConfiguration
        """
        agent_ids = self.get_agent_ids()
        agents = {}

        for agent_id in agent_ids:
            try:
                agents[agent_id] = self.load_agent(agent_id)
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Failed to load agent {agent_id}: {e}")
                continue

        logger.info(f"Loaded {len(agents)}/{len(agent_ids)} agents")
        return agents

    def validate_agent(self, agent_id: str) -> Dict[str, Any]:
        """
        Validate an agent configuration against Truth Protocol

        Args:
            agent_id: Agent ID to validate

        Returns:
            Dict with validation results
        """
        agent = self.load_agent(agent_id)
        validation = {
            'agent_id': agent_id,
            'valid': True,
            'errors': [],
            'warnings': []
        }

        # Validate Truth Protocol compliance
        tp = agent.truth_protocol_compliance

        required_keys = [
            'never_guess', 'pin_versions', 'cite_standards',
            'no_hardcoded_secrets', 'rbac_enforcement',
            'test_coverage_minimum', 'no_skip_rule', 'error_ledger_required'
        ]

        for key in required_keys:
            if key not in tp:
                validation['errors'].append(f"Missing Truth Protocol key: {key}")
                validation['valid'] = False

        # Validate test coverage requirement
        if tp.get('test_coverage_minimum', 0) < 90:
            validation['warnings'].append(
                f"Test coverage minimum {tp.get('test_coverage_minimum')} < 90%"
            )

        # Validate performance SLOs
        slo = agent.performance_slos
        if slo.p95_latency_ms and slo.p95_latency_ms > 500:
            validation['warnings'].append(
                f"P95 latency {slo.p95_latency_ms}ms > 500ms"
            )

        if slo.secrets_in_repo > 0:
            validation['errors'].append(
                f"Secrets in repo: {slo.secrets_in_repo} (must be 0)"
            )
            validation['valid'] = False

        # Validate orchestration commands
        if not agent.orchestration_commands:
            validation['warnings'].append("No orchestration commands defined")

        logger.info(f"Validation for {agent_id}: {'PASS' if validation['valid'] else 'FAIL'}")
        return validation

    def get_quick_reference(self) -> Dict[str, str]:
        """
        Get quick reference mapping from index

        Returns:
            Dict mapping use case to agent ID
        """
        index = self.load_index()
        return index.get('quick_reference', {})

    def clear_cache(self):
        """Clear the agent configuration cache"""
        self._agents_cache.clear()
        self._index = None
        logger.info("Agent configuration cache cleared")


# Example usage
if __name__ == "__main__":
    # Initialize loader
    loader = AgentConfigLoader()

    # Load index
    print("\n=== Agent Index ===")
    index = loader.load_index()
    print(f"Total agents: {index['total_agents']}")
    print(f"Index version: {index['index_version']}")

    # Get all agent IDs
    print("\n=== Agent IDs ===")
    agent_ids = loader.get_agent_ids()
    for aid in agent_ids:
        print(f"  - {aid}")

    # Load specific agent
    print("\n=== Loading Professors of Code ===")
    agent = loader.load_agent("professors-of-code-001")
    print(f"Agent: {agent.agent_name}")
    print(f"Type: {agent.agent_type.value}")
    print(f"Status: {agent.status.value}")
    print(f"Composition: {agent.composition['primary_ai']} + {agent.composition['secondary_ai']}")

    # Show orchestration commands
    print(f"\nOrchestration Commands ({len(agent.orchestration_commands)}):")
    for cmd_name, cmd in agent.orchestration_commands.items():
        print(f"  - {cmd_name}: {cmd.description}")

    # Show performance SLOs
    print(f"\nPerformance SLOs:")
    print(f"  - P95 Latency: {agent.performance_slos.p95_latency_ms}ms")
    print(f"  - Test Coverage: {agent.performance_slos.test_coverage_percent}%")
    print(f"  - Secrets in Repo: {agent.performance_slos.secrets_in_repo}")

    # Validate agent
    print("\n=== Validation ===")
    validation = loader.validate_agent("professors-of-code-001")
    print(f"Valid: {validation['valid']}")
    if validation['errors']:
        print(f"Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")

    # Quick reference
    print("\n=== Quick Reference ===")
    quick_ref = loader.get_quick_reference()
    for use_case, agent_id in quick_ref.items():
        print(f"  {use_case}: {agent_id}")
