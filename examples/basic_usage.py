#!/usr/bin/env python3
"""
Basic usage examples for torch-image-metrics

This example demonstrates how to use the torch-image-metrics library
for calculating various image quality metrics including PSNR, SSIM, 
LPIPS, and FID.
"""

import torch
import sys
from pathlib import Path
import tempfile
from PIL import Image
import numpy as np

# Add project root to path for development
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch_image_metrics as tim


def basic_metrics_example():
    """Quick API usage example"""
    print("🚀 torch-image-metrics Basic Usage Example")
    print("=" * 50)
    
    # Generate sample images
    torch.manual_seed(42)
    img1 = torch.rand(1, 3, 64, 64)
    img2 = img1 + torch.randn_like(img1) * 0.1
    img2 = torch.clamp(img2, 0, 1)
    
    print(f"Image 1 shape: {img1.shape}")
    print(f"Image 2 shape: {img2.shape}")
    print(f"Image 1 range: [{img1.min():.3f}, {img1.max():.3f}]")
    print(f"Image 2 range: [{img2.min():.3f}, {img2.max():.3f}]")
    
    # Quick API usage
    print("\n📊 Quick API Results:")
    psnr_value = tim.quick_psnr(img1, img2)
    ssim_value = tim.quick_ssim(img1, img2)
    mse_value = tim.quick_mse(img1, img2)
    mae_value = tim.quick_mae(img1, img2)
    
    print(f"  PSNR: {psnr_value:.2f} dB")
    print(f"  SSIM: {ssim_value:.4f}")
    print(f"  MSE:  {mse_value:.6f}")
    print(f"  MAE:  {mae_value:.6f}")


def calculator_example():
    """Calculator usage example"""
    print("\n🔧 Calculator Usage Example")
    print("=" * 50)
    
    # Create Calculator instance
    calc = tim.Calculator(device='cpu')
    print(f"Available metrics: {calc.get_available_metrics()}")
    
    # Generate sample images
    torch.manual_seed(123)
    img1 = torch.rand(1, 3, 128, 128)
    img2 = img1 + torch.randn_like(img1) * 0.05
    img2 = torch.clamp(img2, 0, 1)
    
    # Compute all metrics at once
    all_metrics = calc.compute_all_metrics(img1, img2)
    
    print("\n📊 All Metrics Results:")
    print(f"  PSNR: {all_metrics.psnr_db:.2f} dB")
    print(f"  SSIM: {all_metrics.ssim:.4f}")
    print(f"  MSE:  {all_metrics.mse:.6f}")
    print(f"  MAE:  {all_metrics.mae:.6f}")
    
    # Optional metrics (if available)
    if all_metrics.lpips is not None:
        print(f"  LPIPS: {all_metrics.lpips:.4f}")
    if all_metrics.ssim_improved is not None:
        print(f"  SSIM++: {all_metrics.ssim_improved:.4f}")


def dataset_evaluation_example():
    """Dataset evaluation example"""
    print("\n📁 Dataset Evaluation Example")
    print("=" * 50)
    
    # Create temporary directories with test images
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_dir = temp_path / "test"
        ref_dir = temp_path / "ref"
        test_dir.mkdir()
        ref_dir.mkdir()
        
        print(f"Creating test dataset in {temp_dir}")
        
        # Create test images with different patterns
        for i in range(3):
            # Reference image (gradient)
            gradient = np.linspace(0, 255, 64).reshape(1, -1)
            gradient = np.repeat(gradient, 64, axis=0)
            rgb_array = np.stack([gradient, gradient * 0.8, gradient * 0.6], axis=-1)
            ref_img = Image.fromarray(rgb_array.astype(np.uint8), 'RGB')
            ref_img.save(ref_dir / f"image_{i:03d}.png")
            
            # Test image (gradient + noise)
            noise = np.random.normal(0, 10, rgb_array.shape)
            test_array = np.clip(rgb_array + noise, 0, 255).astype(np.uint8)
            test_img = Image.fromarray(test_array, 'RGB')
            test_img.save(test_dir / f"image_{i:03d}.png")
        
        print(f"Created {len(list(test_dir.glob('*.png')))} test images")
        print(f"Created {len(list(ref_dir.glob('*.png')))} reference images")
        
        # Create Evaluator
        evaluator = tim.Evaluator(
            device='cpu',
            use_lpips=False,  # Set to True if lpips is available
            use_improved_ssim=False,  # Set to True if torchmetrics is available
            use_fid=False,  # Set to True for FID calculation
            batch_size=2
        )
        
        # Evaluate dataset
        print("\n🔄 Running dataset evaluation...")
        results = evaluator.evaluate_dataset(test_dir, ref_dir)
        
        # Display results
        print("\n📊 Dataset Evaluation Results:")
        print(f"  Total images: {results.total_images}")
        print(f"  Individual metrics: {len(results.individual_metrics)}")
        
        # Statistics summary
        stats = results.statistics
        if 'psnr_db' in stats:
            print(f"  PSNR: {stats['psnr_db']['mean']:.2f} ± {stats['psnr_db']['std']:.2f} dB")
        if 'ssim' in stats:
            print(f"  SSIM: {stats['ssim']['mean']:.4f} ± {stats['ssim']['std']:.4f}")
        
        # Print detailed summary
        print("\n📋 Detailed Summary:")
        evaluator.print_summary(results)


def advanced_features_example():
    """Advanced features demonstration"""
    print("\n⚡ Advanced Features Example")
    print("=" * 50)
    
    # ImageMatcher usage
    matcher = tim.ImageMatcher(match_strategy='stem')
    print(f"Image matcher strategy: {matcher.match_strategy}")
    
    # Test with temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        dir1 = temp_path / "dir1"
        dir2 = temp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()
        
        # Create matching images
        test_img = Image.new('RGB', (32, 32), (255, 0, 0))
        test_img.save(dir1 / "test_001.png")
        test_img.save(dir2 / "test_001.png")
        
        # Validate datasets
        is_valid, message = matcher.validate_datasets(dir1, dir2)
        print(f"Dataset validation: {is_valid} - {message}")
        
        # Find image pairs
        pairs = matcher.find_image_pairs(dir1, dir2)
        print(f"Found {len(pairs)} matching pairs")
        
        # Get statistics
        stats = matcher.get_matching_statistics(dir1, dir2)
        print(f"Matching statistics: {stats}")


if __name__ == "__main__":
    basic_metrics_example()
    calculator_example()
    dataset_evaluation_example()
    advanced_features_example()
    
    print("\n✅ All examples completed successfully!")
    print("📖 For more information, visit: https://github.com/mdipcit/torch-image-metrics")