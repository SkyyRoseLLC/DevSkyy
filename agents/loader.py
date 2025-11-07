#!/usr/bin/env python3
"""
Agent Configuration Loader with MCP Efficiency Patterns
Loads and validates agent configurations with 5-minute caching

Truth Protocol Compliance: CLAUDE.md
"""

import json
import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, field_validator, ConfigDict

logger = logging.getLogger(__name__)


class AgentConfigError(Exception):
    """Base exception for agent configuration errors"""
    pass


class ConfigValidationError(AgentConfigError):
    """Configuration validation error"""
    pass


class AgentConfigSchema(BaseModel):
    """Pydantic schema for agent configuration validation"""
    model_config = ConfigDict(extra='forbid', strict=True)

    agent_id: str = Field(..., min_length=1, description="Unique agent identifier")
    agent_type: str = Field(..., min_length=1, description="Agent type classification")
    name: str = Field(..., min_length=1, description="Human-readable agent name")
    capabilities: List[str] = Field(..., min_items=1, description="Agent capabilities")
    priority: int = Field(..., ge=1, le=100, description="Agent priority (1-100)")
    available: bool = Field(default=True, description="Agent availability status")
    description: Optional[str] = Field(default=None, description="Agent description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator('capabilities')
    @classmethod
    def validate_capabilities(cls, value: List[str]) -> List[str]:
        """Validate capabilities are non-empty strings"""
        if not all(isinstance(cap, str) and cap.strip() for cap in value):
            raise ValueError("All capabilities must be non-empty strings")
        return value


@dataclass
class AgentConfig:
    """Agent configuration data class"""
    agent_id: str
    agent_type: str
    name: str
    capabilities: List[str]
    priority: int
    available: bool
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches_capability(self, capability: str) -> bool:
        """Check if agent has specific capability"""
        return capability.lower() in [c.lower() for c in self.capabilities]

    def has_any_capability(self, capabilities: List[str]) -> bool:
        """Check if agent has any of the specified capabilities"""
        agent_caps_lower = [c.lower() for c in self.capabilities]
        return any(cap.lower() in agent_caps_lower for cap in capabilities)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'name': self.name,
            'capabilities': self.capabilities,
            'priority': self.priority,
            'available': self.available,
            'description': self.description,
            'metadata': self.metadata
        }


class AgentConfigCache:
    """Simple time-based cache for agent configurations (5-minute TTL)"""

    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, tuple[AgentConfig, float]] = {}
        self.batch_cache: Dict[str, tuple[List[AgentConfig], float]] = {}

    def get(self, key: str) -> Optional[AgentConfig]:
        """Get cached agent config if not expired"""
        if key not in self.cache:
            return None

        config, timestamp = self.cache[key]
        if time.time() - timestamp > self.ttl_seconds:
            del self.cache[key]
            return None

        return config

    def set(self, key: str, config: AgentConfig) -> None:
        """Cache agent config with current timestamp"""
        self.cache[key] = (config, time.time())

    def get_batch(self, key: str) -> Optional[List[AgentConfig]]:
        """Get cached batch of agent configs if not expired"""
        if key not in self.batch_cache:
            return None

        configs, timestamp = self.batch_cache[key]
        if time.time() - timestamp > self.ttl_seconds:
            del self.batch_cache[key]
            return None

        return configs

    def set_batch(self, key: str, configs: List[AgentConfig]) -> None:
        """Cache batch of agent configs with current timestamp"""
        self.batch_cache[key] = (configs, time.time())

    def clear(self) -> None:
        """Clear all cached data"""
        self.cache.clear()
        self.batch_cache.clear()

    def size(self) -> Dict[str, int]:
        """Get cache size information"""
        return {
            'single_configs': len(self.cache),
            'batch_configs': len(self.batch_cache)
        }


class AgentConfigLoader:
    """
    Loads and validates agent configurations with MCP efficiency patterns

    Features:
    - Pydantic validation for all configs
    - 5-minute caching for performance
    - Batch operations for MCP efficiency
    - Comprehensive error handling
    - Config filtering by capability and priority

    Usage:
        loader = AgentConfigLoader(config_dir="config/agents")
        agent = loader.load_agent("scanner_v2")
        agents = loader.load_batch(["scanner_v2", "fixer_v2"])
        filtered = loader.filter_by_capability("security_scan")
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize agent configuration loader

        Args:
            config_dir: Directory containing agent JSON configs
                       Defaults to config/agents/ relative to project root

        Raises:
            AgentConfigError: If config directory doesn't exist
        """
        if config_dir is None:
            project_root = Path(__file__).parent.parent
            self.config_dir = project_root / "config" / "agents"
        else:
            self.config_dir = Path(config_dir)

        if not self.config_dir.exists():
            raise AgentConfigError(f"Config directory not found: {self.config_dir}")

        self.cache = AgentConfigCache(ttl_seconds=300)
        self._config_files: Dict[str, Path] = {}
        self._discover_configs()

        logger.info(
            f"AgentConfigLoader initialized: {len(self._config_files)} configs found "
            f"at {self.config_dir}"
        )

    def _discover_configs(self) -> None:
        """Discover all JSON config files in config directory"""
        self._config_files.clear()

        if not self.config_dir.exists():
            logger.warning(f"Config directory does not exist: {self.config_dir}")
            return

        for json_file in self.config_dir.glob("*.json"):
            agent_id = json_file.stem
            self._config_files[agent_id] = json_file

        logger.debug(f"Discovered {len(self._config_files)} config files")

    def list_available_agents(self) -> List[str]:
        """
        List all available agent IDs

        Returns:
            List of agent IDs found in config directory
        """
        return sorted(self._config_files.keys())

    def load_agent(self, agent_id: str, use_cache: bool = True) -> AgentConfig:
        """
        Load and validate a single agent configuration

        Args:
            agent_id: Agent identifier (filename without .json extension)
            use_cache: Whether to use cached config if available

        Returns:
            AgentConfig object

        Raises:
            AgentConfigError: If agent config not found
            ConfigValidationError: If config validation fails
        """
        # Check cache first
        if use_cache:
            cached = self.cache.get(agent_id)
            if cached is not None:
                logger.debug(f"Cache hit for agent: {agent_id}")
                return cached

        # Load from file
        if agent_id not in self._config_files:
            raise AgentConfigError(f"Agent config not found: {agent_id}")

        config_path = self._config_files[agent_id]

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            # Validate with Pydantic
            validated = AgentConfigSchema(**raw_data)

            # Convert to AgentConfig dataclass
            agent_config = AgentConfig(
                agent_id=validated.agent_id,
                agent_type=validated.agent_type,
                name=validated.name,
                capabilities=validated.capabilities,
                priority=validated.priority,
                available=validated.available,
                description=validated.description,
                metadata=validated.metadata
            )

            # Cache the result
            self.cache.set(agent_id, agent_config)

            logger.info(f"Loaded agent config: {agent_id} ({agent_config.name})")
            return agent_config

        except json.JSONDecodeError as e:
            raise ConfigValidationError(f"Invalid JSON in {config_path}: {e}")
        except Exception as e:
            raise ConfigValidationError(f"Validation failed for {agent_id}: {e}")

    def load_batch(self, agent_ids: List[str], use_cache: bool = True) -> List[AgentConfig]:
        """
        Load multiple agent configurations efficiently (MCP batch operation)

        Args:
            agent_ids: List of agent identifiers to load
            use_cache: Whether to use cached configs if available

        Returns:
            List of AgentConfig objects (same order as input)

        Raises:
            AgentConfigError: If any agent config not found
            ConfigValidationError: If any config validation fails

        Note:
            This is an MCP efficiency pattern - loads multiple configs
            in a single operation to reduce round-trips
        """
        batch_key = "|".join(sorted(agent_ids))

        # Check batch cache
        if use_cache:
            cached_batch = self.cache.get_batch(batch_key)
            if cached_batch is not None:
                logger.debug(f"Batch cache hit for {len(agent_ids)} agents")
                return cached_batch

        # Load all configs
        configs = []
        errors = []

        for agent_id in agent_ids:
            try:
                config = self.load_agent(agent_id, use_cache=use_cache)
                configs.append(config)
            except (AgentConfigError, ConfigValidationError) as e:
                errors.append(f"{agent_id}: {e}")

        if errors:
            error_msg = "; ".join(errors)
            raise AgentConfigError(f"Batch load failed for some agents: {error_msg}")

        # Cache the batch
        self.cache.set_batch(batch_key, configs)

        logger.info(f"Batch loaded {len(configs)} agent configs")
        return configs

    def load_all(self, use_cache: bool = True) -> Dict[str, AgentConfig]:
        """
        Load all available agent configurations

        Args:
            use_cache: Whether to use cached configs if available

        Returns:
            Dictionary mapping agent_id to AgentConfig
        """
        all_ids = self.list_available_agents()

        if not all_ids:
            logger.warning("No agent configs found")
            return {}

        try:
            configs = self.load_batch(all_ids, use_cache=use_cache)
            return {config.agent_id: config for config in configs}
        except (AgentConfigError, ConfigValidationError) as e:
            # If batch fails, try loading individually
            logger.warning(f"Batch load failed, loading individually: {e}")
            result = {}
            for agent_id in all_ids:
                try:
                    config = self.load_agent(agent_id, use_cache=use_cache)
                    result[agent_id] = config
                except (AgentConfigError, ConfigValidationError) as err:
                    logger.error(f"Failed to load {agent_id}: {err}")
            return result

    def filter_by_capability(
        self,
        capability: str,
        configs: Optional[List[AgentConfig]] = None
    ) -> List[AgentConfig]:
        """
        Filter agents by specific capability

        Args:
            capability: Capability to filter by (case-insensitive)
            configs: List of configs to filter (loads all if None)

        Returns:
            List of AgentConfig objects with matching capability
        """
        if configs is None:
            all_configs = self.load_all()
            configs = list(all_configs.values())

        filtered = [
            config for config in configs
            if config.matches_capability(capability)
        ]

        logger.debug(f"Filtered {len(filtered)}/{len(configs)} agents by capability: {capability}")
        return filtered

    def filter_by_capabilities(
        self,
        capabilities: List[str],
        configs: Optional[List[AgentConfig]] = None,
        match_all: bool = False
    ) -> List[AgentConfig]:
        """
        Filter agents by multiple capabilities

        Args:
            capabilities: List of capabilities to filter by
            configs: List of configs to filter (loads all if None)
            match_all: If True, agent must have ALL capabilities;
                      if False, agent must have ANY capability

        Returns:
            List of AgentConfig objects matching criteria
        """
        if configs is None:
            all_configs = self.load_all()
            configs = list(all_configs.values())

        if match_all:
            filtered = [
                config for config in configs
                if all(config.matches_capability(cap) for cap in capabilities)
            ]
        else:
            filtered = [
                config for config in configs
                if config.has_any_capability(capabilities)
            ]

        match_type = "ALL" if match_all else "ANY"
        logger.debug(
            f"Filtered {len(filtered)}/{len(configs)} agents by {match_type} "
            f"capabilities: {capabilities}"
        )
        return filtered

    def filter_by_priority(
        self,
        min_priority: int,
        max_priority: int = 100,
        configs: Optional[List[AgentConfig]] = None
    ) -> List[AgentConfig]:
        """
        Filter agents by priority range

        Args:
            min_priority: Minimum priority (inclusive)
            max_priority: Maximum priority (inclusive)
            configs: List of configs to filter (loads all if None)

        Returns:
            List of AgentConfig objects within priority range
        """
        if configs is None:
            all_configs = self.load_all()
            configs = list(all_configs.values())

        filtered = [
            config for config in configs
            if min_priority <= config.priority <= max_priority
        ]

        logger.debug(
            f"Filtered {len(filtered)}/{len(configs)} agents by priority "
            f"range: {min_priority}-{max_priority}"
        )
        return filtered

    def filter_by_type(
        self,
        agent_type: str,
        configs: Optional[List[AgentConfig]] = None
    ) -> List[AgentConfig]:
        """
        Filter agents by type

        Args:
            agent_type: Agent type to filter by (case-insensitive)
            configs: List of configs to filter (loads all if None)

        Returns:
            List of AgentConfig objects matching type
        """
        if configs is None:
            all_configs = self.load_all()
            configs = list(all_configs.values())

        agent_type_lower = agent_type.lower()
        filtered = [
            config for config in configs
            if config.agent_type.lower() == agent_type_lower
        ]

        logger.debug(f"Filtered {len(filtered)}/{len(configs)} agents by type: {agent_type}")
        return filtered

    def get_available_agents(
        self,
        configs: Optional[List[AgentConfig]] = None
    ) -> List[AgentConfig]:
        """
        Get only available (not busy/offline) agents

        Args:
            configs: List of configs to filter (loads all if None)

        Returns:
            List of available AgentConfig objects
        """
        if configs is None:
            all_configs = self.load_all()
            configs = list(all_configs.values())

        available = [config for config in configs if config.available]

        logger.debug(f"Found {len(available)}/{len(configs)} available agents")
        return available

    def clear_cache(self) -> None:
        """Clear configuration cache"""
        self.cache.clear()
        logger.info("Agent configuration cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dictionary with cache statistics
        """
        return {
            'cache_ttl_seconds': self.cache.ttl_seconds,
            'cache_size': self.cache.size(),
            'config_files_discovered': len(self._config_files),
            'config_directory': str(self.config_dir)
        }

    def validate_config_file(self, agent_id: str) -> Dict[str, Any]:
        """
        Validate a config file and return validation results

        Args:
            agent_id: Agent identifier to validate

        Returns:
            Dictionary with validation results including errors/warnings
        """
        result = {
            'agent_id': agent_id,
            'valid': False,
            'errors': [],
            'warnings': []
        }

        if agent_id not in self._config_files:
            result['errors'].append(f"Config file not found for agent: {agent_id}")
            return result

        try:
            # Try to load and validate
            config = self.load_agent(agent_id, use_cache=False)
            result['valid'] = True

            # Add warnings for potential issues
            if config.priority < 10:
                result['warnings'].append(f"Low priority value: {config.priority}")

            if not config.capabilities:
                result['warnings'].append("No capabilities defined")

            if not config.available:
                result['warnings'].append("Agent marked as unavailable")

        except ConfigValidationError as e:
            result['errors'].append(f"Validation error: {e}")
        except AgentConfigError as e:
            result['errors'].append(f"Config error: {e}")
        except Exception as e:
            result['errors'].append(f"Unexpected error: {e}")

        return result

    def refresh_configs(self) -> None:
        """
        Refresh config file discovery and clear cache

        Useful when config files are added/removed at runtime
        """
        self._discover_configs()
        self.clear_cache()
        logger.info(f"Refreshed configs: {len(self._config_files)} files discovered")
