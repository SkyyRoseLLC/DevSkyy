
"""
DevSkyy Agent Modules Package

This package contains all the specialized AI agents for comprehensive 
website management, optimization, and monitoring.
"""

__version__ = "2.0.0"
__author__ = "DevSkyy Enhanced Platform"

# Import core modules first (scanner and fixer are essential)
from .scanner import scan_site, scan_agents_only
from .fixer import fix_code

# Try to import agent modules with optional dependency handling
__all__ = ['scan_site', 'scan_agents_only', 'fix_code']

try:
    from .financial_agent import FinancialAgent
    __all__.append('FinancialAgent')
except ImportError:
    pass

try:
    from .ecommerce_agent import EcommerceAgent
    __all__.append('EcommerceAgent')
except ImportError:
    pass

try:
    from .wordpress_agent import WordPressAgent
    __all__.append('WordPressAgent')
except ImportError:
    pass

try:
    from .web_development_agent import WebDevelopmentAgent
    __all__.append('WebDevelopmentAgent')
except ImportError:
    pass

try:
    from .site_communication_agent import SiteCommunicationAgent
    __all__.append('SiteCommunicationAgent')
except ImportError:
    pass

try:
    from .brand_intelligence_agent import BrandIntelligenceAgent
    __all__.append('BrandIntelligenceAgent')
except ImportError:
    pass

# Skip modules with heavy dependencies like cv2, complex ML libraries
# These can be imported directly when needed
