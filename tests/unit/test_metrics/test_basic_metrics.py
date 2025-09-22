"""Tests for basic image quality metrics.

This module tests the fundamental metrics: PSNR, SSIM, MSE, MAE, and their
unified interface BasicImageMetrics.
"""

import pytest
import torch
import math

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root / "src"))

from torch_image_metrics.metrics.basic import PSNR, SSIM, MSE, MAE, BasicImageMetrics


class TestPSNR:
    """Test suite for PSNR metric."""
    
    def test_psnr_identical_images(self):
        """Test PSNR returns infinity for identical images."""
        psnr = PSNR(device='cpu')
        img = torch.ones(3, 64, 64)
        
        result = psnr.calculate(img, img)
        assert result == float('inf')
    
    def test_psnr_different_images(self):
        """Test PSNR calculation for different images."""
        psnr = PSNR(device='cpu')
        img1 = torch.zeros(3, 64, 64)
        img2 = torch.ones(3, 64, 64)
        
        result = psnr.calculate(img1, img2)
        # PSNR = 10 * log10(1 / MSE)
        # MSE = 1.0 for this case, so PSNR = 10 * log10(1) = 0
        assert result == 0.0
    
    def test_psnr_with_batch_dimension(self):
        """Test PSNR with batch dimension."""
        psnr = PSNR(device='cpu')
        img1 = torch.rand(2, 3, 32, 32)
        img2 = torch.rand(2, 3, 32, 32)
        
        result = psnr.calculate(img1, img2)
        assert isinstance(result, float)
        assert result > 0  # Should be positive for random images
    
    def test_psnr_known_value(self):
        """Test PSNR calculation with known values."""
        psnr = PSNR(device='cpu')
        img1 = torch.zeros(1, 1, 2, 2)
        img2 = torch.ones(1, 1, 2, 2) * 0.5
        
        # MSE = 0.25, PSNR = 10 * log10(1 / 0.25) = 10 * log10(4) ≈ 6.02
        result = psnr.calculate(img1, img2)
        expected = 10 * math.log10(1 / 0.25)
        assert abs(result - expected) < 1e-5


class TestSSIM:
    """Test suite for SSIM metric."""
    
    def test_ssim_identical_images(self):
        """Test SSIM returns 1.0 for identical images."""
        ssim = SSIM(device='cpu')
        img = torch.rand(3, 64, 64)
        
        result = ssim.calculate(img, img)
        assert abs(result - 1.0) < 0.01  # Allow small numerical error
    
    def test_ssim_different_images(self):
        """Test SSIM calculation for different images."""
        ssim = SSIM(device='cpu')
        img1 = torch.rand(3, 64, 64)
        img2 = torch.rand(3, 64, 64)
        
        result = ssim.calculate(img1, img2)
        assert 0 <= result <= 1  # SSIM should be in [0, 1]
        assert isinstance(result, float)
    
    def test_ssim_with_batch_dimension(self):
        """Test SSIM with batch dimension."""
        ssim = SSIM(device='cpu')
        img1 = torch.rand(2, 3, 64, 64)
        img2 = torch.rand(2, 3, 64, 64)
        
        result = ssim.calculate(img1, img2)
        assert 0 <= result <= 1
        assert isinstance(result, float)
    
    def test_ssim_window_size_validation(self):
        """Test SSIM window size validation."""
        with pytest.raises(ValueError, match="Window size must be odd"):
            SSIM(device='cpu', window_size=10)  # Even window size
    
    def test_ssim_custom_parameters(self):
        """Test SSIM with custom window size and sigma."""
        ssim = SSIM(device='cpu', window_size=7, sigma=2.0)
        img1 = torch.rand(3, 32, 32)
        img2 = torch.rand(3, 32, 32)
        
        result = ssim.calculate(img1, img2)
        assert 0 <= result <= 1


class TestMSE:
    """Test suite for MSE metric."""
    
    def test_mse_identical_images(self):
        """Test MSE returns 0.0 for identical images."""
        mse = MSE(device='cpu')
        img = torch.rand(3, 64, 64)
        
        result = mse.calculate(img, img)
        assert result == 0.0
    
    def test_mse_known_value(self):
        """Test MSE calculation with known values."""
        mse = MSE(device='cpu')
        img1 = torch.zeros(1, 1, 2, 2)
        img2 = torch.ones(1, 1, 2, 2) * 0.5
        
        # MSE = mean((0 - 0.5)^2) = 0.25
        result = mse.calculate(img1, img2)
        assert abs(result - 0.25) < 1e-6
    
    def test_mse_different_images(self):
        """Test MSE calculation for different images."""
        mse = MSE(device='cpu')
        img1 = torch.rand(3, 64, 64)
        img2 = torch.rand(3, 64, 64)
        
        result = mse.calculate(img1, img2)
        assert result >= 0  # MSE is always non-negative
        assert isinstance(result, float)


class TestMAE:
    """Test suite for MAE metric."""
    
    def test_mae_identical_images(self):
        """Test MAE returns 0.0 for identical images."""
        mae = MAE(device='cpu')
        img = torch.rand(3, 64, 64)
        
        result = mae.calculate(img, img)
        assert result == 0.0
    
    def test_mae_known_value(self):
        """Test MAE calculation with known values."""
        mae = MAE(device='cpu')
        img1 = torch.zeros(1, 1, 2, 2)
        img2 = torch.ones(1, 1, 2, 2) * 0.5
        
        # MAE = mean(|0 - 0.5|) = 0.5
        result = mae.calculate(img1, img2)
        assert abs(result - 0.5) < 1e-6
    
    def test_mae_different_images(self):
        """Test MAE calculation for different images."""
        mae = MAE(device='cpu')
        img1 = torch.rand(3, 64, 64)
        img2 = torch.rand(3, 64, 64)
        
        result = mae.calculate(img1, img2)
        assert result >= 0  # MAE is always non-negative
        assert isinstance(result, float)


class TestBasicImageMetrics:
    """Test suite for BasicImageMetrics unified interface."""
    
    def test_calculate_all_returns_dict(self):
        """Test that calculate_all returns a dictionary with all metrics."""
        metrics = BasicImageMetrics(device='cpu')
        img1 = torch.rand(3, 64, 64)
        img2 = torch.rand(3, 64, 64)
        
        result = metrics.calculate_all(img1, img2)
        
        assert isinstance(result, dict)
        assert 'psnr_db' in result
        assert 'ssim' in result
        assert 'mse' in result
        assert 'mae' in result
        
        # Check types
        assert isinstance(result['psnr_db'], float)
        assert isinstance(result['ssim'], float)
        assert isinstance(result['mse'], float)
        assert isinstance(result['mae'], float)
    
    def test_individual_metric_methods(self):
        """Test individual metric computation methods."""
        metrics = BasicImageMetrics(device='cpu')
        img1 = torch.rand(3, 32, 32)
        img2 = torch.rand(3, 32, 32)
        
        psnr_val = metrics.calculate_psnr(img1, img2)
        ssim_val = metrics.calculate_ssim(img1, img2)
        mse_val = metrics.calculate_mse(img1, img2)
        mae_val = metrics.calculate_mae(img1, img2)
        
        # Compare with all metrics result
        all_results = metrics.calculate_all(img1, img2)
        
        assert psnr_val == all_results['psnr_db']
        assert ssim_val == all_results['ssim']
        assert mse_val == all_results['mse']
        assert mae_val == all_results['mae']
    
    def test_custom_ssim_parameters(self):
        """Test BasicImageMetrics with custom SSIM parameters."""
        metrics = BasicImageMetrics(device='cpu', ssim_window_size=7, ssim_sigma=2.0)
        img1 = torch.rand(3, 32, 32)
        img2 = torch.rand(3, 32, 32)
        
        result = metrics.calculate_all(img1, img2)
        assert 'ssim' in result
        assert 0 <= result['ssim'] <= 1
    
    def test_batch_consistency(self):
        """Test that batch and individual processing give consistent results."""
        metrics = BasicImageMetrics(device='cpu')
        
        # Single image pair
        img1 = torch.rand(1, 3, 32, 32)
        img2 = torch.rand(1, 3, 32, 32)
        
        single_result = metrics.calculate_all(img1, img2)
        
        # Same images processed individually  
        individual_psnr = metrics.calculate_psnr(img1, img2)
        individual_ssim = metrics.calculate_ssim(img1, img2)
        
        assert abs(single_result['psnr_db'] - individual_psnr) < 1e-6
        assert abs(single_result['ssim'] - individual_ssim) < 1e-5


if __name__ == '__main__':
    pytest.main([__file__])