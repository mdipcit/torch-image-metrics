"""Basic image quality metrics implementation.

This module implements fundamental image quality metrics including PSNR, SSIM,
MSE, and MAE. These metrics provide essential quality assessment capabilities
for image comparison tasks.

All metrics inherit from BaseMetric and follow the unified interface pattern.
"""

import torch
import torch.nn.functional as F
from typing import Dict
from ..core.base_metric import BaseMetric


class PSNR(BaseMetric):
    """Peak Signal-to-Noise Ratio metric implementation.
    
    PSNR measures the ratio between the maximum possible power of a signal
    and the power of corrupting noise that affects the fidelity of its
    representation. Higher values indicate better image quality.
    
    Formula: PSNR = 10 * log10(1 / MSE)
    
    Args:
        device: Device for computation ('cuda' or 'cpu')
        
    Example:
        >>> psnr = PSNR(device='cuda')
        >>> img1 = torch.rand(1, 3, 64, 64)
        >>> img2 = torch.rand(1, 3, 64, 64)
        >>> score = psnr.calculate(img1, img2)
    """
    
    def _setup_metric_specific_resources(self) -> None:
        """PSNR requires no additional resources."""
        pass
    
    def calculate(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate PSNR between two images.
        
        Args:
            img1: First image tensor, shape (B,C,H,W) or (C,H,W), range [0,1]
            img2: Second image tensor, same shape as img1, range [0,1]
            
        Returns:
            float: PSNR value in dB. Returns inf for identical images.
            
        Raises:
            ValueError: If input tensors have different shapes
        """
        self._validate_input_tensors(img1, img2)
        
        # Move tensors to device
        img1 = self.to_device(img1)
        img2 = self.to_device(img2)
        
        # Calculate MSE
        mse = F.mse_loss(img1, img2)
        
        # Handle identical images
        if mse == 0:
            return float('inf')
        
        # Calculate PSNR
        psnr = 10 * torch.log10(1.0 / mse)
        return psnr.item()


class SSIM(BaseMetric):
    """Structural Similarity Index Measure implementation.
    
    SSIM compares images based on three components: luminance, contrast, and
    structure. It provides a perceptually relevant quality measure that
    correlates better with human visual perception than MSE-based metrics.
    
    Args:
        device: Device for computation ('cuda' or 'cpu')
        window_size: Size of the sliding window (default: 11)
        sigma: Standard deviation for Gaussian kernel (default: 1.5)
        
    Example:
        >>> ssim = SSIM(device='cuda', window_size=11)
        >>> img1 = torch.rand(1, 3, 64, 64) 
        >>> img2 = torch.rand(1, 3, 64, 64)
        >>> score = ssim.calculate(img1, img2)
    """
    
    def __init__(self, device: str = 'cuda', window_size: int = 11, sigma: float = 1.5):
        """Initialize SSIM metric with window parameters.
        
        Args:
            device: Target device for computation
            window_size: Size of the Gaussian window (must be odd)
            sigma: Standard deviation for Gaussian kernel
        """
        self.window_size = window_size
        self.sigma = sigma
        self._window = None
        
        if window_size % 2 == 0:
            raise ValueError("Window size must be odd")
        
        super().__init__(device)
    
    def _setup_metric_specific_resources(self) -> None:
        """Initialize the Gaussian window for SSIM computation."""
        self._window = self._create_gaussian_window(
            self.window_size, self.sigma
        ).to(self.device)
    
    def _create_gaussian_window(self, window_size: int, sigma: float) -> torch.Tensor:
        """Create a 2D Gaussian window for SSIM computation.
        
        Args:
            window_size: Size of the window (odd number)
            sigma: Standard deviation of the Gaussian
            
        Returns:
            torch.Tensor: 2D Gaussian window normalized to sum to 1
        """
        # Create 1D Gaussian kernel
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2
        
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g /= g.sum()
        
        # Create 2D window by outer product
        window_2d = g[None, :] * g[:, None]
        window_2d /= window_2d.sum()
        
        return window_2d
    
    def _apply_gaussian_filter(self, x: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian filter to input tensor.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            window: Gaussian window of shape (window_size, window_size)
            
        Returns:
            torch.Tensor: Filtered tensor
        """
        batch_size, channels, height, width = x.shape
        
        # Expand window to match input channels
        window = window.expand(channels, 1, window.size(0), window.size(1))
        
        # Apply convolution with groups=channels for per-channel filtering
        padding = self.window_size // 2
        filtered = F.conv2d(x, window, padding=padding, groups=channels)
        
        return filtered
    
    def calculate(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate SSIM between two images.
        
        Args:
            img1: First image tensor, shape (B,C,H,W) or (C,H,W), range [0,1]
            img2: Second image tensor, same shape as img1, range [0,1]
            
        Returns:
            float: SSIM value in range [0, 1], where 1 indicates identical images
            
        Raises:
            ValueError: If input tensors have different shapes
        """
        self._validate_input_tensors(img1, img2)
        
        # Add batch dimension if needed
        if len(img1.shape) == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
        
        # Move tensors to device
        img1 = self.to_device(img1)
        img2 = self.to_device(img2)
        
        # SSIM constants
        C1 = 0.01**2
        C2 = 0.03**2
        
        # Calculate local means
        mu1 = self._apply_gaussian_filter(img1, self._window)
        mu2 = self._apply_gaussian_filter(img2, self._window)
        
        # Calculate squared means and cross-correlation
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        
        # Calculate variances and covariance
        sigma1_sq = self._apply_gaussian_filter(img1 * img1, self._window) - mu1_sq
        sigma2_sq = self._apply_gaussian_filter(img2 * img2, self._window) - mu2_sq
        sigma12 = self._apply_gaussian_filter(img1 * img2, self._window) - mu1_mu2
        
        # Calculate SSIM components
        numerator1 = 2 * mu1_mu2 + C1
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2
        
        # Calculate SSIM map
        ssim_map = (numerator1 * numerator2) / (denominator1 * denominator2)
        
        # Return mean SSIM value
        return ssim_map.mean().item()


class MSE(BaseMetric):
    """Mean Squared Error metric implementation.
    
    MSE calculates the average of squared differences between corresponding
    pixels of two images. Lower values indicate better image quality.
    
    Args:
        device: Device for computation ('cuda' or 'cpu')
        
    Example:
        >>> mse = MSE(device='cuda')
        >>> img1 = torch.rand(1, 3, 64, 64)
        >>> img2 = torch.rand(1, 3, 64, 64) 
        >>> score = mse.calculate(img1, img2)
    """
    
    def _setup_metric_specific_resources(self) -> None:
        """MSE requires no additional resources."""
        pass
    
    def calculate(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate MSE between two images.
        
        Args:
            img1: First image tensor, shape (B,C,H,W) or (C,H,W), range [0,1]
            img2: Second image tensor, same shape as img1, range [0,1]
            
        Returns:
            float: MSE value (lower is better)
            
        Raises:
            ValueError: If input tensors have different shapes
        """
        self._validate_input_tensors(img1, img2)
        
        # Move tensors to device
        img1 = self.to_device(img1)
        img2 = self.to_device(img2)
        
        # Calculate MSE
        mse = F.mse_loss(img1, img2)
        return mse.item()


class MAE(BaseMetric):
    """Mean Absolute Error metric implementation.
    
    MAE calculates the average of absolute differences between corresponding
    pixels of two images. Lower values indicate better image quality.
    
    Args:
        device: Device for computation ('cuda' or 'cpu')
        
    Example:
        >>> mae = MAE(device='cuda')
        >>> img1 = torch.rand(1, 3, 64, 64)
        >>> img2 = torch.rand(1, 3, 64, 64)
        >>> score = mae.calculate(img1, img2)
    """
    
    def _setup_metric_specific_resources(self) -> None:
        """MAE requires no additional resources."""
        pass
    
    def calculate(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate MAE between two images.
        
        Args:
            img1: First image tensor, shape (B,C,H,W) or (C,H,W), range [0,1]
            img2: Second image tensor, same shape as img1, range [0,1]
            
        Returns:
            float: MAE value (lower is better)
            
        Raises:
            ValueError: If input tensors have different shapes
        """
        self._validate_input_tensors(img1, img2)
        
        # Move tensors to device
        img1 = self.to_device(img1)
        img2 = self.to_device(img2)
        
        # Calculate MAE  
        mae = F.l1_loss(img1, img2)
        return mae.item()


class BasicImageMetrics:
    """Unified interface for basic image quality metrics.
    
    This class provides a convenient interface to compute all basic metrics
    (PSNR, SSIM, MSE, MAE) using a single instance. It manages the individual
    metric objects and provides batch processing capabilities.
    
    Args:
        device: Device for computation ('cuda' or 'cpu')
        ssim_window_size: Window size for SSIM computation (default: 11)
        ssim_sigma: Sigma for SSIM Gaussian kernel (default: 1.5)
        
    Example:
        >>> metrics = BasicImageMetrics(device='cuda')
        >>> img1 = torch.rand(1, 3, 64, 64)
        >>> img2 = torch.rand(1, 3, 64, 64)
        >>> results = metrics.calculate_all(img1, img2)
        >>> print(f"PSNR: {results['psnr_db']:.2f} dB")
    """
    
    def __init__(self, device: str = 'cuda', ssim_window_size: int = 11, ssim_sigma: float = 1.5):
        """Initialize all basic metrics.
        
        Args:
            device: Target device for computation
            ssim_window_size: Window size for SSIM
            ssim_sigma: Sigma for SSIM Gaussian kernel
        """
        self.device = device
        
        # Initialize individual metrics
        self.psnr = PSNR(device=device)
        self.ssim = SSIM(device=device, window_size=ssim_window_size, sigma=ssim_sigma)
        self.mse = MSE(device=device)
        self.mae = MAE(device=device)
    
    def calculate_all(self, img1: torch.Tensor, img2: torch.Tensor) -> Dict[str, float]:
        """Calculate all basic metrics between two images.
        
        Args:
            img1: First image tensor, shape (B,C,H,W) or (C,H,W), range [0,1]
            img2: Second image tensor, same shape as img1, range [0,1]
            
        Returns:
            Dict[str, float]: Dictionary containing all metric values:
                - psnr_db: PSNR in decibels
                - ssim: SSIM value [0, 1] 
                - mse: Mean Squared Error
                - mae: Mean Absolute Error
                
        Example:
            >>> results = metrics.calculate_all(img1, img2)
            >>> print(f"PSNR: {results['psnr_db']:.2f} dB, SSIM: {results['ssim']:.4f}")
        """
        with torch.no_grad():  # Disable gradient computation for efficiency
            results = {
                'psnr_db': self.psnr.calculate(img1, img2),
                'ssim': self.ssim.calculate(img1, img2),
                'mse': self.mse.calculate(img1, img2),
                'mae': self.mae.calculate(img1, img2)
            }
        
        return results
    
    def calculate_psnr(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate PSNR only."""
        return self.psnr.calculate(img1, img2)
    
    def calculate_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate SSIM only."""
        return self.ssim.calculate(img1, img2)
    
    def calculate_mse(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate MSE only."""
        return self.mse.calculate(img1, img2)
    
    def calculate_mae(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate MAE only."""
        return self.mae.calculate(img1, img2)