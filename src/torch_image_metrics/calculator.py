"""Unified metrics calculator for torch-image-metrics.

This module provides a unified interface for computing all image quality metrics
through a single Calculator class. It integrates basic metrics (PSNR, SSIM),
perceptual metrics (LPIPS), and provides batch processing capabilities.

The Calculator class handles metric initialization, device management, and
graceful degradation when optional dependencies are not available.
"""

import torch
from typing import Dict, List, Optional
from .core.data_structures import IndividualImageMetrics
from .metrics.basic import BasicImageMetrics
from .metrics.perceptual import PerceptualMetrics
from .metrics.dataset import DatasetMetrics


class Calculator:
    """Unified calculator for all image quality metrics.
    
    This class provides a single interface to compute all available metrics
    including basic metrics (PSNR, SSIM, MSE, MAE), perceptual metrics (LPIPS),
    and dataset metrics (FID). It handles device management, optional dependencies,
    and provides both individual and batch processing capabilities.
    
    Args:
        device: Device for computation ('cuda' or 'cpu')
        use_lpips: Whether to enable LPIPS computation (requires lpips package)
        use_improved_ssim: Whether to enable improved SSIM (requires torchmetrics)
        use_fid: Whether to enable FID computation (requires pytorch-fid)
        lpips_net: Network for LPIPS ('alex', 'vgg', 'squeeze')
        fid_batch_size: Batch size for FID computation
        
    Example:
        >>> calc = Calculator(device='cuda')
        >>> img1 = torch.rand(1, 3, 256, 256)
        >>> img2 = torch.rand(1, 3, 256, 256)
        >>> 
        >>> # Compute all metrics for a single pair
        >>> metrics = calc.compute_all_metrics(img1, img2)
        >>> print(f"PSNR: {metrics.psnr_db:.2f} dB")
        >>> 
        >>> # Compute specific metrics only
        >>> psnr = calc.compute_psnr(img1, img2)
        >>> ssim = calc.compute_ssim(img1, img2)
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        use_lpips: bool = True,
        use_improved_ssim: bool = True,
        use_fid: bool = True,
        lpips_net: str = 'alex',
        fid_batch_size: int = 50
    ):
        """Initialize the unified metrics calculator.
        
        Args:
            device: Target device for computation
            use_lpips: Enable LPIPS metric (requires lpips package)
            use_improved_ssim: Enable improved SSIM (requires torchmetrics)
            use_fid: Enable FID metric (requires pytorch-fid) 
            lpips_net: Network architecture for LPIPS
            fid_batch_size: Batch size for FID computation
        """
        self.device = device
        self.available_metrics = ['psnr_db', 'ssim', 'mse', 'mae']  # Always available
        
        # Initialize basic metrics (always available)
        self.basic_metrics = BasicImageMetrics(device=device)
        
        # Initialize perceptual metrics (optional)
        self.perceptual_metrics = PerceptualMetrics(
            device=device,
            use_lpips=use_lpips,
            use_improved_ssim=use_improved_ssim,
            lpips_net=lpips_net
        )
        
        # Add available perceptual metrics to our list
        self.available_metrics.extend(self.perceptual_metrics.get_available_metrics())
        
        # Initialize dataset metrics (optional)
        self.dataset_metrics = DatasetMetrics(
            device=device,
            use_fid=use_fid,
            fid_batch_size=fid_batch_size
        )
        
        # Print initialization summary
        self._print_initialization_summary()
    
    def _print_initialization_summary(self) -> None:
        """Print summary of available metrics."""
        print("torch-image-metrics Calculator initialized")
        print(f"Device: {self.device}")
        print(f"Available metrics: {', '.join(self.available_metrics)}")
        
        # Show what's missing
        missing = []
        if not self.perceptual_metrics.is_available('lpips'):
            missing.append('LPIPS (pip install lpips)')
        if not self.perceptual_metrics.is_available('ssim_improved'):
            missing.append('ImprovedSSIM (pip install torchmetrics)')
        if not self.dataset_metrics.is_available('fid'):
            missing.append('FID (pip install pytorch-fid)')
        
        if missing:
            print(f"Optional metrics not available: {', '.join(missing)}")
    
    def compute_all_metrics(
        self, 
        img1: torch.Tensor, 
        img2: torch.Tensor
    ) -> IndividualImageMetrics:
        """Compute all available metrics for an image pair.
        
        Args:
            img1: First image tensor, shape (B,C,H,W) or (C,H,W), range [0,1]
            img2: Second image tensor, same shape as img1, range [0,1]
            
        Returns:
            IndividualImageMetrics: Complete metrics results with all available values
            
        Example:
            >>> metrics = calc.compute_all_metrics(img1, img2)
            >>> print(f"Basic metrics: {metrics.get_basic_summary()}")
            >>> if metrics.lpips is not None:
            ...     print(f"LPIPS: {metrics.lpips:.4f}")
        """
        with torch.no_grad():  # Disable gradients for efficiency
            # Compute basic metrics (always available)
            basic_results = self.basic_metrics.calculate_all(img1, img2)
            
            # Compute perceptual metrics (optional)
            perceptual_results = self.perceptual_metrics.calculate_all(img1, img2)
            
            # Create IndividualImageMetrics with all results
            metrics = IndividualImageMetrics(
                psnr_db=basic_results['psnr_db'],
                ssim=basic_results['ssim'],
                mse=basic_results['mse'],
                mae=basic_results['mae'],
                lpips=perceptual_results.get('lpips'),
                ssim_improved=perceptual_results.get('ssim_improved')
            )
            
            return metrics
    
    def compute_basic_metrics(
        self, 
        img1: torch.Tensor, 
        img2: torch.Tensor
    ) -> Dict[str, float]:
        """Compute only basic metrics (PSNR, SSIM, MSE, MAE).
        
        Args:
            img1: First image tensor
            img2: Second image tensor
            
        Returns:
            Dict[str, float]: Dictionary with basic metric results
        """
        return self.basic_metrics.calculate_all(img1, img2)
    
    def compute_perceptual_metrics(
        self, 
        img1: torch.Tensor, 
        img2: torch.Tensor
    ) -> Dict[str, float]:
        """Compute only perceptual metrics (LPIPS, ImprovedSSIM).
        
        Args:
            img1: First image tensor
            img2: Second image tensor
            
        Returns:
            Dict[str, float]: Dictionary with available perceptual metric results
        """
        return self.perceptual_metrics.calculate_all(img1, img2)
    
    # Individual metric computation methods
    def compute_psnr(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Compute PSNR only."""
        return self.basic_metrics.calculate_psnr(img1, img2)
    
    def compute_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Compute SSIM only."""
        return self.basic_metrics.calculate_ssim(img1, img2)
    
    def compute_mse(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Compute MSE only."""
        return self.basic_metrics.calculate_mse(img1, img2)
    
    def compute_mae(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Compute MAE only."""
        return self.basic_metrics.calculate_mae(img1, img2)
    
    def compute_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> Optional[float]:
        """Compute LPIPS only (if available)."""
        return self.perceptual_metrics.calculate_lpips(img1, img2)
    
    def compute_improved_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> Optional[float]:
        """Compute ImprovedSSIM only (if available)."""
        return self.perceptual_metrics.calculate_improved_ssim(img1, img2)
    
    def compute_fid(
        self, 
        path1: str, 
        path2: str, 
        verbose: bool = False
    ) -> Optional[float]:
        """Compute FID between two datasets (if available).
        
        Args:
            path1: Path to first dataset directory
            path2: Path to second dataset directory
            verbose: Whether to print progress information
            
        Returns:
            Optional[float]: FID score or None if not available
        """
        return self.dataset_metrics.calculate_fid(path1, path2, verbose=verbose)
    
    def compute_batch_metrics(
        self,
        img_batch1: torch.Tensor,
        img_batch2: torch.Tensor
    ) -> List[IndividualImageMetrics]:
        """Compute metrics for a batch of image pairs.
        
        Args:
            img_batch1: First batch of images, shape (B,C,H,W)
            img_batch2: Second batch of images, shape (B,C,H,W)
            
        Returns:
            List[IndividualImageMetrics]: List of metrics for each image pair
            
        Example:
            >>> batch_size = 10
            >>> batch1 = torch.rand(batch_size, 3, 64, 64)
            >>> batch2 = torch.rand(batch_size, 3, 64, 64)
            >>> results = calc.compute_batch_metrics(batch1, batch2)
            >>> print(f"Processed {len(results)} image pairs")
        """
        if img_batch1.shape[0] != img_batch2.shape[0]:
            raise ValueError("Batch sizes must match")
        
        results = []
        batch_size = img_batch1.shape[0]
        
        with torch.no_grad():
            for i in range(batch_size):
                metrics = self.compute_all_metrics(
                    img_batch1[i:i+1], 
                    img_batch2[i:i+1]
                )
                results.append(metrics)
        
        return results
    
    def compute_batch_statistics(
        self,
        img_batch1: torch.Tensor,
        img_batch2: torch.Tensor
    ) -> Dict[str, Dict[str, float]]:
        """Compute statistical summaries for a batch of image pairs.
        
        Args:
            img_batch1: First batch of images, shape (B,C,H,W)
            img_batch2: Second batch of images, shape (B,C,H,W)
            
        Returns:
            Dict[str, Dict[str, float]]: Statistics (mean, std, min, max) for each metric
            
        Example:
            >>> stats = calc.compute_batch_statistics(batch1, batch2)
            >>> print(f"PSNR: {stats['psnr_db']['mean']:.2f} ± {stats['psnr_db']['std']:.2f}")
        """
        # Compute metrics for all pairs
        results = self.compute_batch_metrics(img_batch1, img_batch2)
        
        # Extract values for each metric
        metric_values = {}
        for metric_name in ['psnr_db', 'ssim', 'mse', 'mae']:
            values = [getattr(result, metric_name) for result in results]
            metric_values[metric_name] = torch.tensor(values)
        
        # Handle optional metrics
        lpips_values = [result.lpips for result in results if result.lpips is not None]
        if lpips_values:
            metric_values['lpips'] = torch.tensor(lpips_values)
        
        ssim_improved_values = [result.ssim_improved for result in results if result.ssim_improved is not None]
        if ssim_improved_values:
            metric_values['ssim_improved'] = torch.tensor(ssim_improved_values)
        
        # Compute statistics
        statistics = {}
        for metric_name, values in metric_values.items():
            statistics[metric_name] = {
                'mean': values.mean().item(),
                'std': values.std().item(),
                'min': values.min().item(),
                'max': values.max().item(),
                'median': values.median().item()
            }
        
        return statistics
    
    def get_available_metrics(self) -> List[str]:
        """Get list of all available metrics."""
        return self.available_metrics.copy()
    
    def is_available(self, metric_name: str) -> bool:
        """Check if a specific metric is available."""
        return metric_name in self.available_metrics
    
    def get_device(self) -> str:
        """Get the current computation device."""
        return self.device
    
    def get_summary(self) -> str:
        """Get a summary of the calculator configuration."""
        lines = [
            "torch-image-metrics Calculator Summary",
            f"Device: {self.device}",
            f"Available metrics: {len(self.available_metrics)}",
            "",
            "Basic metrics (always available):",
            "  - PSNR (Peak Signal-to-Noise Ratio)",
            "  - SSIM (Structural Similarity Index)",
            "  - MSE (Mean Squared Error)", 
            "  - MAE (Mean Absolute Error)",
            "",
        ]
        
        # Perceptual metrics
        perceptual_available = self.perceptual_metrics.get_available_metrics()
        if perceptual_available:
            lines.extend([
                "Perceptual metrics (available):",
                *[f"  - {metric.upper()}" for metric in perceptual_available],
                "",
            ])
        
        # Dataset metrics  
        dataset_available = self.dataset_metrics.get_available_metrics()
        if dataset_available:
            lines.extend([
                "Dataset metrics (available):",
                *[f"  - {metric.upper()}" for metric in dataset_available],
                "",
            ])
        
        # Missing metrics
        all_possible = ['lpips', 'ssim_improved', 'fid']
        missing = [m for m in all_possible if not self.is_available(m)]
        if missing:
            lines.extend([
                "Optional metrics (not available):",
                *[f"  - {metric.upper()}" for metric in missing],
            ])
        
        return "\n".join(lines)