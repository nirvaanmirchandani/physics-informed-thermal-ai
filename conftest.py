"""
Pytest configuration and fixtures.
"""

import sys
from pathlib import Path

# Add src directory to path for test imports
sys.path.insert(0, str(Path(__file__).parent / "src"))
