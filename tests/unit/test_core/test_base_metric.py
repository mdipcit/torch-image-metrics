"""Tests for BaseMetric class.

This module tests the foundational BaseMetric class that all metrics inherit from,
including device management, input validation, and common functionality.
"""

import pytest
import torch
from unittest.mock import Mock

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root / "src"))

from torch_image_metrics.core.base_metric import BaseMetric


class TestMetricImpl(BaseMetric):
    """Test implementation of BaseMetric for testing purposes."""
    
    def _setup_metric_specific_resources(self):
        self.initialized = True
    
    def calculate(self, img1, img2):
        return 0.5


class TestBaseMetric:
    """Test suite for BaseMetric class."""
    
    def test_device_validation_cuda_available(self):
        """Test device validation when CUDA is available."""
        if torch.cuda.is_available():
            metric = TestMetricImpl(device='cuda')
            assert metric.device == 'cuda'
    
    def test_device_validation_cuda_unavailable(self, monkeypatch):
        """Test device validation falls back to CPU when CUDA unavailable."""
        # Mock cuda unavailable
        monkeypatch.setattr(torch, 'cuda', Mock())
        torch.cuda.is_available.return_value = False
        
        metric = TestMetricImpl(device='cuda')
        assert metric.device == 'cpu'
    
    def test_device_validation_cpu_explicit(self):
        """Test explicit CPU device selection."""
        metric = TestMetricImpl(device='cpu')
        assert metric.device == 'cpu'
    
    def test_to_device_moves_tensor(self):
        """Test that to_device moves tensor to correct device."""
        metric = TestMetricImpl(device='cpu')
        tensor = torch.rand(3, 64, 64)
        
        moved = metric.to_device(tensor)
        assert moved.device.type == 'cpu'
    
    def test_validate_input_tensors_same_shape(self):
        """Test input validation with matching shapes."""
        metric = TestMetricImpl(device='cpu')
        img1 = torch.rand(3, 64, 64)
        img2 = torch.rand(3, 64, 64)
        
        # Should not raise
        metric._validate_input_tensors(img1, img2)
    
    def test_validate_input_tensors_different_shapes(self):
        """Test input validation with mismatched shapes."""
        metric = TestMetricImpl(device='cpu')
        img1 = torch.rand(3, 64, 64)
        img2 = torch.rand(3, 32, 32)
        
        with pytest.raises(ValueError, match="Input tensors must have the same shape"):
            metric._validate_input_tensors(img1, img2)
    
    def test_validate_input_tensors_invalid_dimensions(self):
        """Test input validation with invalid tensor dimensions."""
        metric = TestMetricImpl(device='cpu')
        img1 = torch.rand(64, 64)  # 2D instead of 3D/4D
        img2 = torch.rand(64, 64)
        
        with pytest.raises(ValueError, match="Input tensors must be 3D \\(C,H,W\\) or 4D \\(B,C,H,W\\)"):
            metric._validate_input_tensors(img1, img2)
    
    def test_metric_specific_resources_called(self):
        """Test that metric-specific initialization is called."""
        metric = TestMetricImpl(device='cpu')
        assert hasattr(metric, 'initialized')
        assert metric.initialized is True
    
    def test_repr(self):
        """Test string representation of the metric."""
        metric = TestMetricImpl(device='cpu')
        repr_str = repr(metric)
        assert "TestMetricImpl" in repr_str
        assert "device='cpu'" in repr_str
    
    def test_calculate_abstract_method(self):
        """Test that calculate method is abstract."""
        # This test verifies the abstract method works through our test implementation
        metric = TestMetricImpl(device='cpu')
        img1 = torch.rand(3, 64, 64)
        img2 = torch.rand(3, 64, 64)
        
        result = metric.calculate(img1, img2)
        assert result == 0.5


class TestBaseMetricAbstractBehavior:
    """Test abstract behavior of BaseMetric."""
    
    def test_cannot_instantiate_base_metric_directly(self):
        """Test that BaseMetric cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseMetric(device='cpu')
    
    def test_missing_abstract_methods_raises_error(self):
        """Test that missing abstract method implementations raise errors."""
        
        class IncompleteMetric(BaseMetric):
            def _setup_metric_specific_resources(self):
                pass
            # Missing calculate method
        
        with pytest.raises(TypeError):
            IncompleteMetric(device='cpu')


if __name__ == '__main__':
    pytest.main([__file__])