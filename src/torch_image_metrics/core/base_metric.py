"""Base metric class for all image quality metrics.

This module provides the foundational BaseMetric class that all specific
metrics implementations inherit from. It provides unified device management,
error handling, and common interface patterns.
"""

from abc import ABC, abstractmethod
from typing import Union, Dict
import torch


class BaseMetric(ABC):
    """Base class for all image quality metrics.
    
    This abstract base class provides common functionality for all metrics:
    - Unified device management (automatic CUDA/CPU fallback)
    - Common initialization interface
    - Error handling patterns
    - Device validation and setup
    
    All metric implementations should inherit from this class and implement
    the abstract methods.
    
    Args:
        device: Device to use for computation ('cuda' or 'cpu').
               Automatically falls back to CPU if CUDA is not available.
    
    Example:
        >>> class CustomMetric(BaseMetric):
        ...     def _setup_metric_specific_resources(self):
        ...         pass
        ...     
        ...     def calculate(self, img1, img2):
        ...         return 1.0
        >>> 
        >>> metric = CustomMetric(device='cuda')
        >>> result = metric.calculate(torch.randn(1,3,64,64), torch.randn(1,3,64,64))
    """
    
    def __init__(self, device: str = 'cuda'):
        """Initialize the base metric.
        
        Args:
            device: Target device for computation. Defaults to 'cuda'.
        """
        self.device = self._validate_and_setup_device(device)
        self._setup_metric_specific_resources()
    
    def _validate_and_setup_device(self, device: str) -> str:
        """Validate and configure the computation device.
        
        This method replaces the vae-toolkit.DeviceManager functionality
        by providing CUDA availability detection and automatic fallback.
        
        Args:
            device: Requested device ('cuda' or 'cpu')
            
        Returns:
            str: The actual device to be used
            
        Note:
            If CUDA is requested but not available, automatically falls back
            to CPU with a warning message.
        """
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            return 'cpu'
        return device
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to the configured device.
        
        Args:
            tensor: Input tensor to move
            
        Returns:
            torch.Tensor: Tensor moved to the target device
        """
        return tensor.to(self.device)
    
    def _validate_input_tensors(self, img1: torch.Tensor, img2: torch.Tensor) -> None:
        """Validate input tensor shapes and properties.
        
        Args:
            img1: First image tensor
            img2: Second image tensor
            
        Raises:
            ValueError: If tensors have mismatched shapes or invalid properties
        """
        if img1.shape != img2.shape:
            raise ValueError(
                f"Input tensors must have the same shape. "
                f"Got {img1.shape} and {img2.shape}"
            )
        
        if len(img1.shape) not in [3, 4]:
            raise ValueError(
                f"Input tensors must be 3D (C,H,W) or 4D (B,C,H,W). "
                f"Got shape {img1.shape}"
            )
    
    @abstractmethod
    def calculate(
        self, 
        img1: torch.Tensor, 
        img2: torch.Tensor
    ) -> Union[float, Dict[str, float]]:
        """Calculate the metric between two images.
        
        This is the main interface method that all metrics must implement.
        
        Args:
            img1: First image tensor, expected shape (B,C,H,W) or (C,H,W)
            img2: Second image tensor, same shape as img1
            
        Returns:
            Union[float, Dict[str, float]]: Metric value(s). Simple metrics
            return a single float, complex metrics may return a dictionary
            of multiple values.
            
        Raises:
            ValueError: If input tensors are invalid
            RuntimeError: If computation fails
        """
        pass
    
    @abstractmethod
    def _setup_metric_specific_resources(self) -> None:
        """Initialize metric-specific resources.
        
        This method is called during __init__ and should handle any
        metric-specific initialization such as loading models, setting up
        loss functions, etc.
        
        Each metric implementation should override this method to perform
        its own initialization tasks.
        """
        pass
    
    def __repr__(self) -> str:
        """String representation of the metric."""
        return f"{self.__class__.__name__}(device='{self.device}')"