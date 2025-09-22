#!/usr/bin/env python3
"""
Basic tests for torch-image-metrics package

These tests verify the basic functionality and imports of the package.
"""

import pytest
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def test_package_imports():
    """Test that the package can be imported successfully"""
    try:
        import src.torch_image_metrics as tim
        assert hasattr(tim, '__version__')
        assert tim.__version__ == "0.1.0"
    except ImportError:
        pytest.fail("Failed to import torch_image_metrics package")


def test_package_metadata():
    """Test package metadata"""
    import src.torch_image_metrics as tim
    
    assert tim.__author__ == "Yus314"
    assert tim.__email__ == "shizhaoyoujie@gmail.com"
    assert tim.__version__ == "0.1.0"


def test_submodule_imports():
    """Test that submodules can be imported (even if empty for now)"""
    try:
        import src.torch_image_metrics.core
        import src.torch_image_metrics.metrics
        import src.torch_image_metrics.utils
    except ImportError as e:
        pytest.fail(f"Failed to import submodules: {e}")


if __name__ == "__main__":
    pytest.main([__file__])