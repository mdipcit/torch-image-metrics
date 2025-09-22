"""Perceptual image quality metrics implementation.

This module implements advanced perceptual metrics that better correlate with
human visual perception, including LPIPS (Learned Perceptual Image Patch
Similarity) and improved SSIM using torchmetrics.

These metrics require additional dependencies and will gracefully degrade
when packages are not available.
"""

import torch
import warnings
from typing import Optional, Dict
from ..core.base_metric import BaseMetric


class LPIPS(BaseMetric):
    """Learned Perceptual Image Patch Similarity metric implementation.
    
    LPIPS computes perceptual distance between images using deep features
    from pre-trained networks. It correlates better with human perceptual
    judgments than traditional pixel-based metrics.
    
    Args:
        device: Device for computation ('cuda' or 'cpu')
        net: Network to use ('alex', 'vgg', or 'squeeze') - default: 'alex'
        verbose: Whether to print loading information - default: False
        
    Requires:
        lpips package: pip install lpips
        
    Example:
        >>> try:
        ...     lpips = LPIPS(device='cuda')
        ...     img1 = torch.rand(1, 3, 64, 64)
        ...     img2 = torch.rand(1, 3, 64, 64)
        ...     score = lpips.calculate(img1, img2)
        ... except ImportError:
        ...     print("LPIPS requires 'pip install lpips'")
    """
    
    def __init__(self, device: str = 'cuda', net: str = 'alex', verbose: bool = False):
        """Initialize LPIPS metric.
        
        Args:
            device: Target device for computation
            net: Pre-trained network ('alex', 'vgg', 'squeeze')
            verbose: Print loading information
            
        Raises:
            ImportError: If lpips package is not installed
        """
        self.net = net
        self.verbose = verbose
        self.loss_fn = None
        
        super().__init__(device)
    
    def _setup_metric_specific_resources(self) -> None:
        """Initialize LPIPS network.
        
        Raises:
            ImportError: If lpips package is not available
        """
        try:
            import lpips
            
            # Initialize LPIPS model
            self.loss_fn = lpips.LPIPS(net=self.net, verbose=self.verbose)
            
            # Move to device
            self.loss_fn = self.loss_fn.to(self.device)
            
            # Set to evaluation mode
            self.loss_fn.eval()
            
        except ImportError as e:
            raise ImportError(
                "LPIPS requires the 'lpips' package. "
                "Install it with: pip install lpips"
            ) from e
    
    def _normalize_for_lpips(self, img: torch.Tensor) -> torch.Tensor:
        """Normalize image tensor for LPIPS computation.
        
        LPIPS expects images in range [-1, 1], while our interface uses [0, 1].
        
        Args:
            img: Input image tensor in range [0, 1]
            
        Returns:
            torch.Tensor: Normalized tensor in range [-1, 1]
        """
        # Check if image is already in [-1, 1] range
        if img.min() >= -1.0 and img.max() <= 1.0 and img.min() < 0:
            return img
        
        # Convert from [0, 1] to [-1, 1]
        return img * 2.0 - 1.0
    
    def calculate(self, img1: torch.Tensor, img2: torch.Tensor) -> Optional[float]:
        """Calculate LPIPS between two images.
        
        Args:
            img1: First image tensor, shape (B,C,H,W) or (C,H,W), range [0,1]
            img2: Second image tensor, same shape as img1, range [0,1]
            
        Returns:
            Optional[float]: LPIPS distance (lower is better), or None if calculation fails
            
        Raises:
            ValueError: If input tensors have different shapes
            RuntimeError: If LPIPS computation fails
        """
        self._validate_input_tensors(img1, img2)
        
        # Add batch dimension if needed
        if len(img1.shape) == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
        
        try:
            # Move tensors to device and normalize
            img1 = self.to_device(img1)
            img2 = self.to_device(img2)
            
            img1_norm = self._normalize_for_lpips(img1)
            img2_norm = self._normalize_for_lpips(img2)
            
            # Compute LPIPS with no gradient tracking
            with torch.no_grad():
                distance = self.loss_fn(img1_norm, img2_norm)
                return distance.mean().item()
                
        except Exception as e:
            warnings.warn(f"LPIPS calculation failed: {e}")
            return None


class ImprovedSSIM(BaseMetric):
    """Improved SSIM implementation using TorchMetrics.
    
    This implementation uses the TorchMetrics library's SSIM implementation,
    which may provide better numerical stability and additional features
    compared to the basic SSIM implementation.
    
    Args:
        device: Device for computation ('cuda' or 'cpu')
        data_range: Range of the image values - default: 1.0
        kernel_size: Size of the averaging kernel - default: 11
        sigma: Standard deviation for Gaussian kernel - default: 1.5
        
    Requires:
        torchmetrics package: pip install torchmetrics
        
    Example:
        >>> try:
        ...     ssim = ImprovedSSIM(device='cuda')
        ...     img1 = torch.rand(1, 3, 64, 64)
        ...     img2 = torch.rand(1, 3, 64, 64)
        ...     score = ssim.calculate(img1, img2)
        ... except ImportError:
        ...     print("ImprovedSSIM requires 'pip install torchmetrics'")
    """
    
    def __init__(
        self, 
        device: str = 'cuda',
        data_range: float = 1.0,
        kernel_size: int = 11,
        sigma: float = 1.5
    ):
        """Initialize ImprovedSSIM metric.
        
        Args:
            device: Target device for computation
            data_range: Range of image values (1.0 for [0,1] range)
            kernel_size: Size of the averaging kernel (must be odd)
            sigma: Standard deviation for Gaussian kernel
            
        Raises:
            ImportError: If torchmetrics package is not installed
        """
        self.data_range = data_range
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.ssim_metric = None
        
        super().__init__(device)
    
    def _setup_metric_specific_resources(self) -> None:
        """Initialize TorchMetrics SSIM.
        
        Raises:
            ImportError: If torchmetrics package is not available
        """
        try:
            from torchmetrics.image import StructuralSimilarityIndexMeasure
            
            # Initialize SSIM metric
            self.ssim_metric = StructuralSimilarityIndexMeasure(
                data_range=self.data_range,
                kernel_size=self.kernel_size,
                sigma=self.sigma
            )
            
            # Move to device
            self.ssim_metric = self.ssim_metric.to(self.device)
            
        except ImportError as e:
            raise ImportError(
                "ImprovedSSIM requires the 'torchmetrics' package. "
                "Install it with: pip install torchmetrics"
            ) from e
    
    def calculate(self, img1: torch.Tensor, img2: torch.Tensor) -> Optional[float]:
        """Calculate improved SSIM between two images.
        
        Args:
            img1: First image tensor, shape (B,C,H,W) or (C,H,W), range [0,1]
            img2: Second image tensor, same shape as img1, range [0,1]
            
        Returns:
            Optional[float]: SSIM value in range [0, 1], or None if calculation fails
            
        Raises:
            ValueError: If input tensors have different shapes
        """
        self._validate_input_tensors(img1, img2)
        
        # Add batch dimension if needed
        if len(img1.shape) == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
        
        try:
            # Move tensors to device
            img1 = self.to_device(img1)
            img2 = self.to_device(img2)
            
            # Reset metric state
            self.ssim_metric.reset()
            
            # Compute SSIM
            with torch.no_grad():
                ssim_value = self.ssim_metric(img1, img2)
                return ssim_value.item()
                
        except Exception as e:
            warnings.warn(f"ImprovedSSIM calculation failed: {e}")
            return None


class PerceptualMetrics:
    """Unified interface for perceptual image quality metrics.
    
    This class provides a convenient interface to compute perceptual metrics
    (LPIPS, ImprovedSSIM) with graceful degradation when packages are unavailable.
    
    Args:
        device: Device for computation ('cuda' or 'cpu')
        use_lpips: Whether to initialize LPIPS (default: True)
        use_improved_ssim: Whether to initialize improved SSIM (default: True)
        lpips_net: Network for LPIPS ('alex', 'vgg', 'squeeze')
        
    Example:
        >>> metrics = PerceptualMetrics(device='cuda')
        >>> img1 = torch.rand(1, 3, 64, 64)
        >>> img2 = torch.rand(1, 3, 64, 64)
        >>> results = metrics.calculate_all(img1, img2)
        >>> if 'lpips' in results:
        ...     print(f"LPIPS: {results['lpips']:.4f}")
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        use_lpips: bool = True,
        use_improved_ssim: bool = True,
        lpips_net: str = 'alex'
    ):
        """Initialize perceptual metrics.
        
        Args:
            device: Target device for computation
            use_lpips: Whether to attempt LPIPS initialization
            use_improved_ssim: Whether to attempt ImprovedSSIM initialization
            lpips_net: Network for LPIPS computation
        """
        self.device = device
        self.lpips = None
        self.improved_ssim = None
        self.available_metrics = []
        
        # Initialize LPIPS if requested
        if use_lpips:
            try:
                self.lpips = LPIPS(device=device, net=lpips_net)
                self.available_metrics.append('lpips')
            except ImportError:
                warnings.warn(
                    "LPIPS not available. Install with: pip install lpips"
                )
        
        # Initialize ImprovedSSIM if requested
        if use_improved_ssim:
            try:
                self.improved_ssim = ImprovedSSIM(device=device)
                self.available_metrics.append('ssim_improved')
            except ImportError:
                warnings.warn(
                    "ImprovedSSIM not available. Install with: pip install torchmetrics"
                )
    
    def calculate_all(self, img1: torch.Tensor, img2: torch.Tensor) -> Dict[str, float]:
        """Calculate all available perceptual metrics.
        
        Args:
            img1: First image tensor, shape (B,C,H,W) or (C,H,W), range [0,1]
            img2: Second image tensor, same shape as img1, range [0,1]
            
        Returns:
            Dict[str, float]: Dictionary containing available metric values.
                Keys may include: 'lpips', 'ssim_improved'
        """
        results = {}
        
        # Calculate LPIPS if available
        if self.lpips is not None:
            lpips_value = self.lpips.calculate(img1, img2)
            if lpips_value is not None:
                results['lpips'] = lpips_value
        
        # Calculate ImprovedSSIM if available
        if self.improved_ssim is not None:
            ssim_value = self.improved_ssim.calculate(img1, img2)
            if ssim_value is not None:
                results['ssim_improved'] = ssim_value
        
        return results
    
    def calculate_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> Optional[float]:
        """Calculate LPIPS only."""
        if self.lpips is not None:
            return self.lpips.calculate(img1, img2)
        return None
    
    def calculate_improved_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> Optional[float]:
        """Calculate ImprovedSSIM only."""
        if self.improved_ssim is not None:
            return self.improved_ssim.calculate(img1, img2)
        return None
    
    def get_available_metrics(self) -> list:
        """Get list of available metrics."""
        return self.available_metrics.copy()
    
    def is_available(self, metric_name: str) -> bool:
        """Check if a specific metric is available."""
        return metric_name in self.available_metrics