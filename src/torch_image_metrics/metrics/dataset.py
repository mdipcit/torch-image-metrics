"""Dataset-level image quality metrics implementation.

This module implements metrics that evaluate entire datasets rather than
individual image pairs. These metrics are particularly useful for evaluating
generative models and large-scale image processing pipelines.

The primary metric is FID (Fréchet Inception Distance) which compares
the distributions of features extracted from two datasets.
"""

import torch
import warnings
from typing import Optional, Tuple
from pathlib import Path
from ..core.base_metric import BaseMetric


class FID(BaseMetric):
    """Fréchet Inception Distance metric implementation.
    
    FID computes the distance between two multivariate Gaussians fitted to
    feature representations of images extracted using a pre-trained Inception
    network. Lower FID scores indicate better image quality and diversity.
    
    This metric is commonly used for evaluating generative models like GANs
    and diffusion models, as it captures both quality and diversity of
    generated images.
    
    Args:
        device: Device for computation ('cuda' or 'cpu')
        batch_size: Batch size for feature extraction (default: 50)
        
    Requires:
        pytorch-fid package: pip install pytorch-fid
        
    Example:
        >>> try:
        ...     fid = FID(device='cuda')
        ...     score = fid.calculate_dataset('/path/to/real', '/path/to/fake')
        ...     print(f"FID Score: {score:.2f}")
        ... except ImportError:
        ...     print("FID requires 'pip install pytorch-fid'")
    """
    
    def __init__(self, device: str = 'cuda', batch_size: int = 50):
        """Initialize FID metric.
        
        Args:
            device: Target device for computation
            batch_size: Batch size for processing images
            
        Raises:
            ImportError: If pytorch-fid package is not installed
        """
        self.batch_size = batch_size
        self.fid_score_module = None
        self.inception_model = None
        
        super().__init__(device)
    
    def _setup_metric_specific_resources(self) -> None:
        """Initialize FID computation resources.
        
        Raises:
            ImportError: If pytorch-fid package is not available
        """
        try:
            # Import pytorch-fid modules
            import pytorch_fid.fid_score as fid_score
            from pytorch_fid.inception import InceptionV3
            
            self.fid_score_module = fid_score
            
            # Initialize Inception model for feature extraction
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            self.inception_model = InceptionV3([block_idx])
            
            # Move model to device and set to evaluation mode
            self.inception_model = self.inception_model.to(self.device)
            self.inception_model.eval()
            
        except ImportError as e:
            raise ImportError(
                "FID requires the 'pytorch-fid' package. "
                "Install it with: pip install pytorch-fid"
            ) from e
    
    def calculate(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """FID cannot be calculated on individual image pairs.
        
        This method is required by BaseMetric but FID is designed for
        dataset-level evaluation. Use calculate_dataset instead.
        
        Raises:
            NotImplementedError: Always, as FID requires datasets not image pairs
        """
        raise NotImplementedError(
            "FID cannot be calculated on individual image pairs. "
            "Use calculate_dataset(path1, path2) instead."
        )
    
    def calculate_dataset(self, path1: str, path2: str, verbose: bool = False) -> Optional[float]:
        """Calculate FID between two image datasets.
        
        Args:
            path1: Path to first dataset directory
            path2: Path to second dataset directory
            verbose: Whether to print progress information
            
        Returns:
            Optional[float]: FID score (lower is better), or None if calculation fails
            
        Raises:
            ValueError: If dataset paths don't exist or contain no images
            RuntimeError: If FID calculation fails
        """
        # Validate paths
        path1 = Path(path1)
        path2 = Path(path2)
        
        if not path1.exists() or not path1.is_dir():
            raise ValueError(f"Dataset path does not exist: {path1}")
        if not path2.exists() or not path2.is_dir():
            raise ValueError(f"Dataset path does not exist: {path2}")
        
        try:
            if verbose:
                print("Calculating FID between:")
                print(f"  Dataset 1: {path1}")
                print(f"  Dataset 2: {path2}")
            
            # Use pytorch-fid's calculate_fid_given_paths function
            fid_value = self.fid_score_module.calculate_fid_given_paths(
                [str(path1), str(path2)],
                batch_size=self.batch_size,
                device=self.device,
                dims=2048
            )
            
            if verbose:
                print(f"FID Score: {fid_value:.4f}")
            
            return fid_value
            
        except Exception as e:
            warnings.warn(f"FID calculation failed: {e}")
            return None
    
    def calculate_fid_from_statistics(
        self, 
        mu1: torch.Tensor, 
        sigma1: torch.Tensor,
        mu2: torch.Tensor, 
        sigma2: torch.Tensor
    ) -> float:
        """Calculate FID from pre-computed statistics.
        
        This method allows for FID calculation when you have already computed
        the mean and covariance of the feature distributions.
        
        Args:
            mu1: Mean of first distribution, shape (2048,)
            sigma1: Covariance of first distribution, shape (2048, 2048)
            mu2: Mean of second distribution, shape (2048,)
            sigma2: Covariance of second distribution, shape (2048, 2048)
            
        Returns:
            float: FID score computed from the statistics
        """
        if self.fid_score_module is None:
            raise RuntimeError("FID module not initialized")
        
        return self.fid_score_module.calculate_frechet_distance(
            mu1.cpu().numpy(), 
            sigma1.cpu().numpy(),
            mu2.cpu().numpy(), 
            sigma2.cpu().numpy()
        )
    
    def compute_statistics(self, path: str, verbose: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute statistics for a dataset.
        
        This method extracts features and computes mean and covariance
        that can be stored and reused for multiple FID calculations.
        
        Args:
            path: Path to dataset directory
            verbose: Whether to print progress information
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and covariance tensors
            
        Raises:
            ValueError: If dataset path doesn't exist
        """
        path = Path(path)
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Dataset path does not exist: {path}")
        
        if verbose:
            print(f"Computing statistics for dataset: {path}")
        
        try:
            # Get activation statistics
            mu, sigma = self.fid_score_module.calculate_activation_statistics(
                str(path),
                self.inception_model,
                batch_size=self.batch_size,
                dims=2048,
                device=self.device
            )
            
            return torch.tensor(mu), torch.tensor(sigma)
            
        except Exception as e:
            raise RuntimeError(f"Failed to compute statistics: {e}") from e


class DatasetMetrics:
    """Unified interface for dataset-level metrics.
    
    This class provides a convenient interface for computing dataset-level
    metrics with graceful degradation when packages are unavailable.
    
    Args:
        device: Device for computation ('cuda' or 'cpu')
        use_fid: Whether to initialize FID (default: True)
        fid_batch_size: Batch size for FID computation
        
    Example:
        >>> metrics = DatasetMetrics(device='cuda')
        >>> results = metrics.calculate_all('/path/to/test', '/path/to/ref')
        >>> if 'fid' in results:
        ...     print(f"FID: {results['fid']:.2f}")
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        use_fid: bool = True,
        fid_batch_size: int = 50
    ):
        """Initialize dataset metrics.
        
        Args:
            device: Target device for computation
            use_fid: Whether to attempt FID initialization
            fid_batch_size: Batch size for FID computation
        """
        self.device = device
        self.fid = None
        self.available_metrics = []
        
        # Initialize FID if requested
        if use_fid:
            try:
                self.fid = FID(device=device, batch_size=fid_batch_size)
                self.available_metrics.append('fid')
            except ImportError:
                warnings.warn(
                    "FID not available. Install with: pip install pytorch-fid"
                )
    
    def calculate_all(
        self, 
        path1: str, 
        path2: str, 
        verbose: bool = False
    ) -> dict:
        """Calculate all available dataset metrics.
        
        Args:
            path1: Path to first dataset directory
            path2: Path to second dataset directory
            verbose: Whether to print progress information
            
        Returns:
            Dict[str, float]: Dictionary containing available metric values
        """
        results = {}
        
        # Calculate FID if available
        if self.fid is not None:
            fid_value = self.fid.calculate_dataset(path1, path2, verbose=verbose)
            if fid_value is not None:
                results['fid'] = fid_value
        
        return results
    
    def calculate_fid(
        self, 
        path1: str, 
        path2: str, 
        verbose: bool = False
    ) -> Optional[float]:
        """Calculate FID only."""
        if self.fid is not None:
            return self.fid.calculate_dataset(path1, path2, verbose=verbose)
        return None
    
    def get_available_metrics(self) -> list:
        """Get list of available metrics."""
        return self.available_metrics.copy()
    
    def is_available(self, metric_name: str) -> bool:
        """Check if a specific metric is available."""
        return metric_name in self.available_metrics