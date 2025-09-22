# torch-image-metrics

[![PyPI version](https://badge.fury.io/py/torch-image-metrics.svg)](https://badge.fury.io/py/torch-image-metrics)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-image-metrics.svg)](https://pypi.org/project/torch-image-metrics/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**torch-image-metrics** is a unified PyTorch library for image quality evaluation, providing implementations of popular metrics like PSNR, SSIM, LPIPS, and FID with a simple, consistent API.

## ✨ Features

- 🚀 **Fast & Efficient**: GPU-accelerated computations with PyTorch
- 📊 **Comprehensive Metrics**: PSNR, SSIM, MSE, MAE, LPIPS, ImprovedSSIM, FID
- 🎯 **Unified API**: Consistent interface across all metrics
- 📁 **Dataset Evaluation**: Bulk evaluation with statistical analysis
- 🔄 **Batch Processing**: Efficient batch computation support
- 🛠️ **Flexible**: Optional dependencies for advanced metrics
- 📈 **Easy Integration**: Drop-in replacement for existing workflows

## 📦 Installation

### Basic Installation
```bash
pip install torch-image-metrics
```

### With Optional Dependencies (Recommended)
```bash
pip install torch-image-metrics[full]
```

### Development Installation
```bash
git clone https://github.com/mdipcit/torch-image-metrics.git
cd torch-image-metrics
pip install -e .[dev]
```

## 🚀 Quick Start

### Quick API (Single Metrics)

```python
import torch
import torch_image_metrics as tim

# Generate sample images
img1 = torch.rand(1, 3, 256, 256)  # Reference image
img2 = torch.rand(1, 3, 256, 256)  # Test image

# Calculate individual metrics
psnr = tim.quick_psnr(img1, img2)      # Peak Signal-to-Noise Ratio
ssim = tim.quick_ssim(img1, img2)      # Structural Similarity Index
mse = tim.quick_mse(img1, img2)        # Mean Squared Error
mae = tim.quick_mae(img1, img2)        # Mean Absolute Error

print(f"PSNR: {psnr:.2f} dB")
print(f"SSIM: {ssim:.4f}")
```

### Calculator API (Multiple Metrics)

```python
import torch_image_metrics as tim

# Initialize calculator
calc = tim.Calculator(device='cuda')  # or 'cpu'

# Compute all available metrics at once
metrics = calc.compute_all_metrics(img1, img2)

print(f"PSNR: {metrics.psnr_db:.2f} dB")
print(f"SSIM: {metrics.ssim:.4f}")
print(f"MSE: {metrics.mse:.6f}")
print(f"MAE: {metrics.mae:.6f}")

# Optional metrics (if dependencies available)
if metrics.lpips is not None:
    print(f"LPIPS: {metrics.lpips:.4f}")
if metrics.ssim_improved is not None:
    print(f"SSIM++: {metrics.ssim_improved:.4f}")
```

### Dataset Evaluation

```python
import torch_image_metrics as tim
from pathlib import Path

# Initialize evaluator with desired metrics
evaluator = tim.Evaluator(
    device='cuda',
    use_lpips=True,           # Enable LPIPS (requires lpips package)
    use_improved_ssim=True,   # Enable SSIM++ (requires torchmetrics)
    use_fid=True,             # Enable FID (requires pytorch-fid)
    batch_size=16
)

# Evaluate entire datasets
test_dir = Path("path/to/test/images")
ref_dir = Path("path/to/reference/images")

results = evaluator.evaluate_dataset(test_dir, ref_dir)

# Access results
print(f"Total images evaluated: {results.total_images}")
print(f"FID Score: {results.fid_score:.2f}")

# Statistical summary
stats = results.statistics
print(f"PSNR: {stats['psnr_db']['mean']:.2f} ± {stats['psnr_db']['std']:.2f} dB")
print(f"SSIM: {stats['ssim']['mean']:.4f} ± {stats['ssim']['std']:.4f}")

# Print comprehensive summary
evaluator.print_summary(results)
```

## 📊 Supported Metrics

| Metric | Description | Type | Requires |
|--------|-------------|------|----------|
| **PSNR** | Peak Signal-to-Noise Ratio | Full-Reference | Core |
| **SSIM** | Structural Similarity Index | Full-Reference | Core |
| **MSE** | Mean Squared Error | Full-Reference | Core |
| **MAE** | Mean Absolute Error | Full-Reference | Core |
| **LPIPS** | Learned Perceptual Image Patch Similarity | Full-Reference | `lpips` package |
| **SSIM++** | Improved Structural Similarity | Full-Reference | `torchmetrics` package |
| **FID** | Fréchet Inception Distance | Dataset-level | `pytorch-fid` package |

## 🔧 Advanced Usage

### Custom Image Matching

```python
import torch_image_metrics as tim

# Initialize image matcher for dataset evaluation
matcher = tim.ImageMatcher(match_strategy='stem')  # or 'full_name'

# Validate dataset structure
is_valid, message = matcher.validate_datasets(test_dir, ref_dir)
if not is_valid:
    print(f"Dataset validation failed: {message}")

# Find matching image pairs
pairs = matcher.find_image_pairs(test_dir, ref_dir)
print(f"Found {len(pairs)} matching pairs")

# Get detailed statistics
stats = matcher.get_matching_statistics(test_dir, ref_dir)
print(f"Matching statistics: {stats}")
```

### Batch Processing

```python
import torch
import torch_image_metrics as tim

# Process batches of images efficiently
batch_size = 8
calc = tim.Calculator(device='cuda')

test_batch = torch.rand(batch_size, 3, 256, 256)
ref_batch = torch.rand(batch_size, 3, 256, 256)

# Batch computation for better performance
for i in range(batch_size):
    metrics = calc.compute_all_metrics(
        test_batch[i:i+1], 
        ref_batch[i:i+1]
    )
    print(f"Image {i}: PSNR={metrics.psnr_db:.2f} dB")
```

## 🔧 Configuration

### Optional Dependencies

torch-image-metrics gracefully handles optional dependencies:

- **LPIPS**: Install with `pip install lpips`
- **ImprovedSSIM**: Install with `pip install torchmetrics`
- **FID**: Install with `pip install pytorch-fid`

If optional dependencies are not available, the corresponding metrics will return `None` values.

### Device Management

```python
import torch_image_metrics as tim

# Automatic device detection
calc = tim.Calculator()  # Uses CUDA if available, else CPU

# Explicit device specification
calc_gpu = tim.Calculator(device='cuda')
calc_cpu = tim.Calculator(device='cpu')

# Device verification
print(f"Using device: {calc.device}")
```

## 📈 Performance Tips

1. **Use GPU**: Enable CUDA for significant speedup
2. **Batch Processing**: Process multiple images together when possible
3. **Appropriate Image Size**: Resize large images for faster computation
4. **Disable Unused Metrics**: Turn off expensive metrics like LPIPS/FID if not needed

```python
# Performance-optimized evaluator
evaluator = tim.Evaluator(
    device='cuda',
    use_lpips=False,      # Disable for speed
    use_fid=False,        # Disable for speed
    batch_size=32,        # Larger batches
    image_size=256        # Resize for speed
)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/mdipcit/torch-image-metrics.git
cd torch-image-metrics

# Install with development dependencies
pip install -e .[dev]

# Run tests
pytest

# Code quality checks
ruff check src/ --fix
ruff format src/
mypy src/torch_image_metrics/
```

## 📚 Documentation

- **API Reference**: [Coming Soon]
- **Examples**: See the [`examples/`](examples/) directory
- **Migration Guide**: [Coming Soon]

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/torch_image_metrics --cov-report=term-missing

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
```

## 🛡️ Requirements

- **Python**: 3.10 or 3.11
- **PyTorch**: ≥2.0.0
- **torchvision**: ≥0.15.0
- **Pillow**: ≥9.0.0
- **NumPy**: ≥1.21.0

### Optional Dependencies

- **lpips**: ≥0.1.4 (for LPIPS metric)
- **pytorch-fid**: ≥0.3.0 (for FID metric)
- **torchmetrics**: ≥1.8.2 (for ImprovedSSIM metric)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This library was developed as part of the [Generative-Latent-Optimization](https://github.com/mdipcit/Generative-Latent-Optimization) project and extracted into a standalone package for broader use.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/mdipcit/torch-image-metrics/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mdipcit/torch-image-metrics/discussions)

---

**Made with ❤️ for the computer vision community**