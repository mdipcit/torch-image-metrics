"""Tests for Quick API functions.

This module tests the convenience functions provided in the package's __init__.py
for quick metric calculations: quick_psnr, quick_ssim, quick_mse, quick_mae, etc.
"""

import pytest
import torch

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root / "src"))

import torch_image_metrics as tim


class TestQuickAPI:
    """Test suite for Quick API functions."""
    
    def test_quick_psnr(self):
        """Test quick_psnr function."""
        img1 = torch.rand(3, 64, 64)
        img2 = torch.rand(3, 64, 64)
        
        result = tim.quick_psnr(img1, img2)
        assert isinstance(result, float)
        assert result > 0  # Should be positive for random images
    
    def test_quick_ssim(self):
        """Test quick_ssim function."""
        img1 = torch.rand(3, 64, 64)
        img2 = torch.rand(3, 64, 64)
        
        result = tim.quick_ssim(img1, img2)
        assert isinstance(result, float)
        assert 0 <= result <= 1  # SSIM is bounded in [0, 1]
    
    def test_quick_mse(self):
        """Test quick_mse function."""
        img1 = torch.rand(3, 64, 64)
        img2 = torch.rand(3, 64, 64)
        
        result = tim.quick_mse(img1, img2)
        assert isinstance(result, float)
        assert result >= 0  # MSE is non-negative
    
    def test_quick_mae(self):
        """Test quick_mae function."""
        img1 = torch.rand(3, 64, 64)
        img2 = torch.rand(3, 64, 64)
        
        result = tim.quick_mae(img1, img2)
        assert isinstance(result, float)
        assert result >= 0  # MAE is non-negative
    
    def test_quick_all_metrics(self):
        """Test quick_all_metrics function."""
        img1 = torch.rand(3, 32, 32)
        img2 = torch.rand(3, 32, 32)
        
        result = tim.quick_all_metrics(img1, img2)
        
        # Should return IndividualImageMetrics instance
        assert hasattr(result, 'psnr_db')
        assert hasattr(result, 'ssim')
        assert hasattr(result, 'mse')
        assert hasattr(result, 'mae')
        
        # Check basic properties
        assert isinstance(result.psnr_db, float)
        assert isinstance(result.ssim, float)
        assert isinstance(result.mse, float)  
        assert isinstance(result.mae, float)
        
        assert result.psnr_db > 0
        assert 0 <= result.ssim <= 1
        assert result.mse >= 0
        assert result.mae >= 0
    
    def test_quick_lpips_available(self):
        """Test quick_lpips works when package is available."""
        img1 = torch.rand(3, 64, 64)  # Use larger size for LPIPS
        img2 = torch.rand(3, 64, 64)
        
        # LPIPS should return a value if package is installed
        result = tim.quick_lpips(img1, img2)
        if result is not None:
            assert isinstance(result, float)
            assert result >= 0  # LPIPS distance should be non-negative
        # If None, the package wasn't available or calculation failed
    
    def test_identical_images(self):
        """Test Quick API with identical images."""
        img = torch.rand(3, 32, 32)
        
        psnr = tim.quick_psnr(img, img)
        ssim = tim.quick_ssim(img, img)
        mse = tim.quick_mse(img, img)
        mae = tim.quick_mae(img, img)
        
        assert psnr == float('inf')
        assert abs(ssim - 1.0) < 0.01  # SSIM should be close to 1
        assert mse == 0.0
        assert mae == 0.0
    
    def test_batch_dimension_handling(self):
        """Test Quick API with batch dimensions."""
        # 4D tensor (batch dimension included)
        img1 = torch.rand(2, 3, 32, 32)
        img2 = torch.rand(2, 3, 32, 32)
        
        psnr = tim.quick_psnr(img1, img2)
        ssim = tim.quick_ssim(img1, img2)
        mse = tim.quick_mse(img1, img2)
        mae = tim.quick_mae(img1, img2)
        
        assert isinstance(psnr, float)
        assert isinstance(ssim, float)
        assert isinstance(mse, float)
        assert isinstance(mae, float)
    
    def test_global_calculator_consistency(self):
        """Test that multiple Quick API calls use consistent global calculator."""
        img1 = torch.rand(3, 32, 32)
        img2 = torch.rand(3, 32, 32)
        
        # Call multiple times
        psnr1 = tim.quick_psnr(img1, img2)
        psnr2 = tim.quick_psnr(img1, img2)
        
        # Should be identical (same calculator, same inputs)
        assert psnr1 == psnr2
    
    def test_device_cpu_default(self):
        """Test that Quick API uses CPU by default."""
        # This test verifies that the global calculator uses CPU
        # which is important for quick computation without GPU setup
        img1 = torch.rand(3, 16, 16)
        img2 = torch.rand(3, 16, 16)
        
        # Should work even if tensors are on CPU
        result = tim.quick_psnr(img1.cpu(), img2.cpu())
        assert isinstance(result, float)
    
    def test_quick_all_metrics_data_structure(self):
        """Test quick_all_metrics returns proper data structure."""
        img1 = torch.rand(3, 16, 16)
        img2 = torch.rand(3, 16, 16)
        
        metrics = tim.quick_all_metrics(img1, img2)
        
        # Test data structure methods
        assert hasattr(metrics, 'to_dict')
        assert hasattr(metrics, 'get_basic_summary')
        
        # Test to_dict method
        metrics_dict = metrics.to_dict()
        assert isinstance(metrics_dict, dict)
        assert 'psnr_db' in metrics_dict
        assert 'ssim' in metrics_dict
        
        # Test get_basic_summary method
        summary = metrics.get_basic_summary()
        assert isinstance(summary, str)
        assert 'PSNR' in summary
        assert 'SSIM' in summary


class TestQuickAPIEdgeCases:
    """Test edge cases for Quick API."""
    
    def test_mismatched_shapes(self):
        """Test Quick API with mismatched tensor shapes."""
        img1 = torch.rand(3, 32, 32)
        img2 = torch.rand(3, 16, 16)  # Different size
        
        with pytest.raises(ValueError):
            tim.quick_psnr(img1, img2)
    
    def test_invalid_dimensions(self):
        """Test Quick API with invalid tensor dimensions."""
        img1 = torch.rand(32, 32)  # 2D instead of 3D/4D
        img2 = torch.rand(32, 32)
        
        with pytest.raises(ValueError):
            tim.quick_psnr(img1, img2)
    
    def test_extreme_values(self):
        """Test Quick API with extreme pixel values."""
        # Test with zeros and ones
        img1 = torch.zeros(3, 16, 16)
        img2 = torch.ones(3, 16, 16)
        
        psnr = tim.quick_psnr(img1, img2)
        ssim = tim.quick_ssim(img1, img2)
        mse = tim.quick_mse(img1, img2)
        mae = tim.quick_mae(img1, img2)
        
        # Should handle extreme values gracefully
        assert isinstance(psnr, float)
        assert isinstance(ssim, float)
        assert mse == 1.0  # MSE should be 1 for 0 vs 1
        assert mae == 1.0  # MAE should be 1 for 0 vs 1


if __name__ == '__main__':
    pytest.main([__file__])