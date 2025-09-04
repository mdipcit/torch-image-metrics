# Package Name

[![PyPI version](https://badge.fury.io/py/package-name.svg)](https://badge.fury.io/py/package-name)
[![Python Version](https://img.shields.io/pypi/pyversions/package-name.svg)](https://pypi.org/project/package-name/)

A brief description of your package.

## Installation

```bash
pip install package-name
```

## Usage

```python
import package_name

# Your usage example here
```

## Development

```bash
# Clone the repository
git clone https://github.com/Yus314/package-name.git
cd package-name

# Install in development mode
pip install -e .
```

## Publishing

Publishing is automated via GitHub Actions. To publish a new version:

1. Update the version in `pyproject.toml`
2. Create a git tag:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
3. GitHub Actions will automatically build and publish to PyPI

## License

MIT