#!/usr/bin/env python3
"""
Test fixtures for agent tests
Isolated from main conftest to avoid import issues
"""

import sys
from pathlib import Path

# Add project root to path BEFORE any other imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
