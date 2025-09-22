"""torch-image-metrics: Unified PyTorch image quality metrics library

A comprehensive library for calculating various image quality metrics 
including PSNR, SSIM, LPIPS, and FID.

Usage:
    Basic usage:
        import torch_image_metrics as tim
        
        psnr_value = tim.quick_psnr(img1, img2)
        ssim_value = tim.quick_ssim(img1, img2)
    
    Advanced usage:
        calc = tim.Calculator(device='cuda')
        metrics = calc.compute_all_metrics(img1, img2)
        
        evaluator = tim.Evaluator(metrics=['psnr', 'ssim', 'lpips'])
        results = evaluator.evaluate_dataset(test_dir, ref_dir)
"""

import torch
from typing import Optional

__version__ = "0.1.0"
__author__ = "Yus314"
__email__ = "shizhaoyoujie@gmail.com"

# Import core components
from .calculator import Calculator
from .evaluator import Evaluator
from .core.data_structures import IndividualImageMetrics, AllMetricsResults

# Import utilities
from .utils.image_matcher import ImageMatcher

# Import metric classes for advanced usage
from .metrics import (
    PSNR, SSIM, MSE, MAE, BasicImageMetrics,
    LPIPS, ImprovedSSIM, PerceptualMetrics,
    FID, DatasetMetrics
)

# Global calculator instance for quick API
_global_calculator = None

def _get_global_calculator():
    """Get or create global calculator instance for quick API."""
    global _global_calculator
    if _global_calculator is None:
        _global_calculator = Calculator(device='cpu')  # Use CPU by default for quick API
    return _global_calculator

# Quick API functions
def quick_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Quick PSNR calculation using CPU.
    
    Args:
        img1: First image tensor, shape (B,C,H,W) or (C,H,W), range [0,1]
        img2: Second image tensor, same shape as img1, range [0,1]
        
    Returns:
        float: PSNR value in dB
        
    Example:
        >>> import torch
        >>> import torch_image_metrics as tim
        >>> img1 = torch.rand(3, 64, 64)
        >>> img2 = torch.rand(3, 64, 64)
        >>> psnr = tim.quick_psnr(img1, img2)
        >>> print(f"PSNR: {psnr:.2f} dB")
    """
    calc = _get_global_calculator()
    return calc.compute_psnr(img1, img2)

def quick_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Quick SSIM calculation using CPU.
    
    Args:
        img1: First image tensor, shape (B,C,H,W) or (C,H,W), range [0,1]
        img2: Second image tensor, same shape as img1, range [0,1]
        
    Returns:
        float: SSIM value in range [0, 1]
        
    Example:
        >>> import torch
        >>> import torch_image_metrics as tim
        >>> img1 = torch.rand(3, 64, 64)
        >>> img2 = torch.rand(3, 64, 64)
        >>> ssim = tim.quick_ssim(img1, img2)
        >>> print(f"SSIM: {ssim:.4f}")
    """
    calc = _get_global_calculator()
    return calc.compute_ssim(img1, img2)

def quick_mse(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Quick MSE calculation using CPU.
    
    Args:
        img1: First image tensor, shape (B,C,H,W) or (C,H,W), range [0,1]
        img2: Second image tensor, same shape as img1, range [0,1]
        
    Returns:
        float: MSE value (lower is better)
    """
    calc = _get_global_calculator()
    return calc.compute_mse(img1, img2)

def quick_mae(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Quick MAE calculation using CPU.
    
    Args:
        img1: First image tensor, shape (B,C,H,W) or (C,H,W), range [0,1]
        img2: Second image tensor, same shape as img1, range [0,1]
        
    Returns:
        float: MAE value (lower is better)
    """
    calc = _get_global_calculator()
    return calc.compute_mae(img1, img2)

def quick_lpips(img1: torch.Tensor, img2: torch.Tensor) -> Optional[float]:
    """Quick LPIPS calculation using CPU (if available).
    
    Args:
        img1: First image tensor, shape (B,C,H,W) or (C,H,W), range [0,1]
        img2: Second image tensor, same shape as img1, range [0,1]
        
    Returns:
        Optional[float]: LPIPS value (lower is better) or None if not available
        
    Note:
        Requires 'pip install lpips' for functionality.
    """
    calc = _get_global_calculator()
    return calc.compute_lpips(img1, img2)

def quick_all_metrics(img1: torch.Tensor, img2: torch.Tensor) -> IndividualImageMetrics:
    """Quick calculation of all available metrics using CPU.
    
    Args:
        img1: First image tensor, shape (B,C,H,W) or (C,H,W), range [0,1]
        img2: Second image tensor, same shape as img1, range [0,1]
        
    Returns:
        IndividualImageMetrics: Complete metrics results
        
    Example:
        >>> import torch
        >>> import torch_image_metrics as tim
        >>> img1 = torch.rand(3, 64, 64)
        >>> img2 = torch.rand(3, 64, 64)
        >>> metrics = tim.quick_all_metrics(img1, img2)
        >>> print(metrics.get_basic_summary())
    """
    calc = _get_global_calculator()
    return calc.compute_all_metrics(img1, img2)

__all__ = [
    # Core classes
    "Calculator",
    "Evaluator",
    
    # Data structures
    "IndividualImageMetrics", 
    "AllMetricsResults",
    
    # Utilities
    "ImageMatcher",
    
    # Individual metric classes
    "PSNR", "SSIM", "MSE", "MAE", "BasicImageMetrics",
    "LPIPS", "ImprovedSSIM", "PerceptualMetrics",
    "FID", "DatasetMetrics",
    
    # Quick API functions
    "quick_psnr",
    "quick_ssim",
    "quick_mse",
    "quick_mae",
    "quick_lpips",
    "quick_all_metrics",
]