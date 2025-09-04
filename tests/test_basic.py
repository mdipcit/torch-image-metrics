"""Basic tests for package_name."""

import package_name


def test_version():
    """Test that version is accessible."""
    assert hasattr(package_name, "__version__")
    assert isinstance(package_name.__version__, str)