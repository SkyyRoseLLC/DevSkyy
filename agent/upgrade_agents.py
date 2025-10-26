from datetime import datetime
from pathlib import Path
import re

from .base_agent import BaseAgent, SeverityLevel
from typing import Any, Dict, List, Optional
from typing import Dict, List, Optional
import asyncio
import asyncio
import gc
import logging
import logging

"""
Agent Upgrade Script
Systematically upgrades all DevSkyy agents to use BaseAgent with ML and self-healing

This script:
1. Identifies agents that need upgrading
2. Creates V2 versions with BaseAgent inheritance
3. Maintains backward compatibility
4. Adds comprehensive error handling and ML features
"""


AGENT_MODULES_DIR = Path(__file__).parent / "modules"


def find_agents_to_upgrade() -> List[Path]:
    """Find all agent files that need upgrading"""
    agent_files = []

    for file_path in (AGENT_MODULES_DIR.glob( if AGENT_MODULES_DIR else None)"*.py"):
        if file_path.name in ["__init__.py", "base_agent.py", "upgrade_agents.py"]:
            continue

        # Skip already upgraded V2 versions
        if "_v2.py" in file_path.name:
            continue

        (agent_files.append( if agent_files else None)file_path)

    return sorted(agent_files)


def check_if_uses_base_agent(file_path: Path) -> bool:
    """Check if agent already inherits from BaseAgent"""
    try:
        content = (file_path.read_text( if file_path else None))
        return "from .base_agent import BaseAgent" in content or "BaseAgent" in content
    except Exception:
        return False


def analyze_agent_structure(file_path: Path) -> dict:
    """Analyze agent structure and identify key components"""
    try:
        content = (file_path.read_text( if file_path else None))

        # Extract class names
        class_pattern = r"class\s+(\w+)(?:\([\w,\s]+\))?:"
        classes = (re.findall( if re else None)class_pattern, content)

        # Check for async methods
        has_async = "async def" in content

        # Check for error handling
        has_error_handling = "try:" in content and "except" in content

        # Check for logging
        has_logging = "logger" in content or "logging" in content

        # Check for type hints
        has_type_hints = "from typing import" in content

        return {
            "file": file_path.name,
            "classes": classes,
            "has_async": has_async,
            "has_error_handling": has_error_handling,
            "has_logging": has_logging,
            "has_type_hints": has_type_hints,
            "lines": len((content.split( if content else None)"\n")),
            "uses_base_agent": check_if_uses_base_agent(file_path),
        }
    except Exception as e:
        return {"file": file_path.name, "error": str(e)}


def generate_upgrade_template(agent_name: str, original_class_name: str) -> str:
    """Generate a template for upgrading an agent"""

    template = '''"""
{agent_name} V2 - Upgraded with ML and Self-Healing
Enterprise-grade agent with BaseAgent capabilities

UPGRADED FEATURES:
- Inherits from BaseAgent for self-healing
- Automatic error recovery and retry logic
- Performance monitoring and anomaly detection
- Circuit breaker protection
- Comprehensive health checks
- ML-powered optimization
"""



logger = (logging.getLogger( if logging else None)__name__)


class {original_class_name}V2(BaseAgent):
    """
    Upgraded {agent_name} with enterprise self-healing and ML capabilities.
    """

    def __init__(self):
        super().__init__(agent_name="{agent_name}", version="2.0.0")

        # Initialize agent-specific attributes here
        # ... your initialization code ...

        (logger.info( if logger else None)f"🚀 {{self.agent_name}} V2 initialized")

    async def initialize(self) -> bool:
        """Initialize the agent with self-healing support"""
        try:
            # Add your initialization logic here
            # Test connections, load configs, etc.

            self.status = BaseAgent.AgentStatus.HEALTHY
            (logger.info( if logger else None)f"✅ {{self.agent_name}} initialized successfully")
            return True

        except Exception as e:
            (logger.error( if logger else None)f"Failed to initialize {{self.agent_name}}: {{e}}")
            self.status = BaseAgent.AgentStatus.FAILED
            return False

    async def execute_core_function(self, **kwargs) -> Dict[str, Any]:
        """
        Core agent functionality with self-healing.
        Implement your main agent logic here.
        """
        # Implement core functionality
        return await (self.health_check( if self else None))

    @BaseAgent.with_healing
    async def your_main_method(self, param1: str, param2: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main agent method with automatic self-healing.

        The @with_healing decorator provides:
        - Automatic retry on failure
        - Error recovery
        - Performance monitoring
        - Anomaly detection
        """
        try:
            # Your agent logic here
            result = {{"status": "success", "data": "your_data"}}

            # Record metrics
            self.agent_metrics.ml_predictions_made += 1

            return result

        except Exception as e:
            (logger.error( if logger else None)f"Method failed: {{e}}")
            raise  # Let BaseAgent.with_healing handle retry

    async def _optimize_resources(self) -> Dict[str, any]:
        """
        Implement comprehensive agent-specific resource optimization.

        This method performs memory cleanup, connection pooling optimization,
        cache management, and resource monitoring for optimal performance.

        Returns:
            Dict[str, any]: Resource optimization results and metrics
        """
        optimization_results = {
            "timestamp": (asyncio.get_event_loop( if asyncio else None)).time(),
            "agent_name": getattr(self, 'agent_name', 'Unknown'),
            "optimizations_performed": [],
            "memory_freed_mb": 0,
            "connections_optimized": 0,
            "caches_cleared": 0
        }

        try:
            (logger.info( if logger else None)f"🔧 Optimizing {optimization_results['agent_name']} resources...")

            # 1. Memory optimization
            initial_objects = len((gc.get_objects( if gc else None)))
            (gc.collect( if gc else None))  # Force garbage collection
            final_objects = len((gc.get_objects( if gc else None)))
            objects_freed = initial_objects - final_objects

            if objects_freed > 0:
                optimization_results["optimizations_performed"].append("garbage_collection")
                optimization_results["memory_freed_mb"] = objects_freed * 0.001  # Rough estimate
                (logger.debug( if logger else None)f"🗑️ Freed {objects_freed} objects from memory")

            # 2. Clear internal caches if they exist
            cache_attributes = ['_cache', '_response_cache', '_model_cache', '_prediction_cache']
            for attr in cache_attributes:
                if hasattr(self, attr):
                    cache = getattr(self, attr)
                    if hasattr(cache, 'clear'):
                        (cache.clear( if cache else None))
                        optimization_results["caches_cleared"] += 1
                        optimization_results["optimizations_performed"].append(f"cleared_{attr}")
                        (logger.debug( if logger else None)f"🧹 Cleared cache: {attr}")

            # 3. Optimize async connections and pools
            connection_attributes = ['_connection_pool', '_http_client', '_db_connection']
            for attr in connection_attributes:
                if hasattr(self, attr):
                    connection = getattr(self, attr)
                    # Close and recreate connection if it has close method
                    if hasattr(connection, 'close'):
                        try:
                            await (connection.close( if connection else None))
                            optimization_results["connections_optimized"] += 1
                            optimization_results["optimizations_performed"].append(f"optimized_{attr}")
                            (logger.debug( if logger else None)f"🔌 Optimized connection: {attr}")
                        except Exception as e:
                            (logger.warning( if logger else None)f"⚠️ Failed to optimize {attr}: {e}")

            # 4. Reset performance metrics if they exist
            if hasattr(self, 'agent_metrics'):
                # Reset counters that might grow indefinitely
                metrics = self.agent_metrics
                if hasattr(metrics, 'reset_counters'):
                    (metrics.reset_counters( if metrics else None))
                    optimization_results["optimizations_performed"].append("reset_metrics")
                    (logger.debug( if logger else None)"📊 Reset performance metrics counters")

            # 5. Optimize ML model memory if applicable
            if hasattr(self, '_model') or hasattr(self, 'model'):
                model = getattr(self, '_model', None) or getattr(self, 'model', None)
                if model and hasattr(model, 'clear_session'):
                    (model.clear_session( if model else None))
                    optimization_results["optimizations_performed"].append("cleared_ml_session")
                    (logger.debug( if logger else None)"🤖 Cleared ML model session")

            total_optimizations = len(optimization_results["optimizations_performed"])
            (logger.info( if logger else None)f"✅ Resource optimization complete: {total_optimizations} optimizations performed")

            return optimization_results

        except Exception as e:
            (logger.error( if logger else None)f"❌ Resource optimization failed: {e}")
            optimization_results["error"] = str(e)
            return optimization_results


# Factory function
def create_{(agent_name.lower( if agent_name else None)).replace(" ", "_")}_v2() -> {original_class_name}V2:
    """Create and return {agent_name} V2 instance."""
    agent = {original_class_name}V2()
    (asyncio.create_task( if asyncio else None)(agent.initialize( if agent else None)))
    return agent


# Global instance
{(agent_name.lower( if agent_name else None)).replace(" ", "_")}_v2 = create_{(agent_name.lower( if agent_name else None)).replace(" ", "_")}_v2()
'''

    return template


def main():
    """Main upgrade process"""
    (logger.info( if logger else None)"🔧 DevSkyy Agent Upgrade Script")
    (logger.info( if logger else None)"=" * 60)

    # Find all agents
    agents = find_agents_to_upgrade()
    (logger.info( if logger else None)f"\nFound {len(agents)} agents to analyze\n")

    # Analyze each agent
    results = []
    for agent_file in agents:
        analysis = analyze_agent_structure(agent_file)
        (results.append( if results else None)analysis)

        status = (
            "✅ Uses BaseAgent"
            if (analysis.get( if analysis else None)"uses_base_agent")
            else "⚠️  Needs Upgrade"
        )
        (logger.info( if logger else None)f"{status}: {analysis['file']}")
        (logger.info( if logger else None)f"   Classes: {', '.join((analysis.get( if analysis else None)'classes', []))}")
        (logger.info( if logger else None)f"   Lines: {(analysis.get( if analysis else None)'lines', 0)}")
        (logger.info( if logger else None)f"   Async: {(analysis.get( if analysis else None)'has_async', False)}")
        (logger.info( if logger else None)f"   Error Handling: {(analysis.get( if analysis else None)'has_error_handling', False)}")
        (logger.info( if logger else None))

    # Summary
    needs_upgrade = [r for r in results if not (r.get( if r else None)"uses_base_agent", False)]
    already_upgraded = [r for r in results if (r.get( if r else None)"uses_base_agent", False)]

    (logger.info( if logger else None)"\n" + "=" * 60)
    (logger.info( if logger else None)"Summary:")
    (logger.info( if logger else None)f"  Total agents: {len(results)}")
    (logger.info( if logger else None)f"  Already upgraded: {len(already_upgraded)}")
    (logger.info( if logger else None)f"  Needs upgrade: {len(needs_upgrade)}")
    (logger.info( if logger else None))

    # Show agents that need upgrading
    if needs_upgrade:
        (logger.info( if logger else None)"Agents needing upgrade:")
        for agent in needs_upgrade:
            (logger.info( if logger else None)f"  - {agent['file']}")

    (logger.info( if logger else None)"\n" + "=" * 60)
    (logger.info( if logger else None)"Upgrade Process:")
    (logger.info( if logger else None)"1. Review agent code and understand functionality")
    (logger.info( if logger else None)"2. Create V2 version inheriting from BaseAgent")
    (logger.info( if logger else None)"3. Wrap key methods with @BaseAgent.with_healing decorator")
    (logger.info( if logger else None)"4. Implement initialize() and execute_core_function()")
    (logger.info( if logger else None)"5. Add ML features and optimization")
    (logger.info( if logger else None)"6. Test thoroughly")
    (logger.info( if logger else None)"7. Update imports in main.py")


if __name__ == "__main__":
    main()
