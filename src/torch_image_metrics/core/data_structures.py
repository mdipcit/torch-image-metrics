"""Data structures for storing image quality metric results.

This module defines the data classes used to store and organize
image quality metric results, both for individual images and
comprehensive dataset evaluations.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime
import json


@dataclass
class IndividualImageMetrics:
    """Container for individual image quality metrics.
    
    This data class stores metric results for a single image comparison,
    including both basic and advanced metrics. Optional metrics are set
    to None when not computed.
    
    Attributes:
        psnr_db: Peak Signal-to-Noise Ratio in decibels
        ssim: Structural Similarity Index Measure (0 to 1)
        mse: Mean Squared Error
        mae: Mean Absolute Error
        lpips: Learned Perceptual Image Patch Similarity (optional)
        ssim_improved: Improved SSIM using torchmetrics (optional)
        
    Example:
        >>> metrics = IndividualImageMetrics(
        ...     psnr_db=25.5,
        ...     ssim=0.85,
        ...     mse=0.001,
        ...     mae=0.02,
        ...     lpips=0.15
        ... )
        >>> print(f"PSNR: {metrics.psnr_db:.2f} dB")
    """
    
    # Basic metrics (always computed)
    psnr_db: float
    ssim: float
    mse: float
    mae: float
    
    # Advanced metrics (optional, computed when requested)
    lpips: Optional[float] = None
    ssim_improved: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format.
        
        Returns:
            Dict[str, Any]: Dictionary representation of metrics
        """
        result = {
            'psnr_db': self.psnr_db,
            'ssim': self.ssim,
            'mse': self.mse,
            'mae': self.mae
        }
        
        if self.lpips is not None:
            result['lpips'] = self.lpips
        if self.ssim_improved is not None:
            result['ssim_improved'] = self.ssim_improved
            
        return result
    
    def get_basic_summary(self) -> str:
        """Get a basic summary string of the metrics.
        
        Returns:
            str: Formatted summary of main metrics
        """
        return f"PSNR: {self.psnr_db:.2f}dB, SSIM: {self.ssim:.4f}, MSE: {self.mse:.6f}"


@dataclass
class AllMetricsResults:
    """Comprehensive container for dataset evaluation results.
    
    This data class stores complete evaluation results including individual
    image metrics, dataset-level metrics (like FID), statistical summaries,
    and metadata about the evaluation.
    
    Attributes:
        individual_metrics: List of metrics for each image pair
        fid_score: Fréchet Inception Distance for the entire dataset
        statistics: Statistical summaries (mean, std, etc.) for each metric
        total_images: Total number of images processed
        evaluation_timestamp: ISO timestamp of when evaluation was performed
        created_dataset_path: Path to the generated/test images
        original_dataset_path: Path to the reference/ground truth images
        
    Example:
        >>> results = AllMetricsResults(
        ...     individual_metrics=[...],
        ...     fid_score=15.2,
        ...     statistics={'psnr': {'mean': 25.5, 'std': 2.1}},
        ...     total_images=1000,
        ...     evaluation_timestamp='2024-01-01T12:00:00Z',
        ...     created_dataset_path='/path/to/test',
        ...     original_dataset_path='/path/to/ref'
        ... )
        >>> print(results.get_metric_summary())
    """
    
    # Core results
    individual_metrics: List[IndividualImageMetrics]
    fid_score: float
    statistics: Dict[str, Dict[str, float]]
    
    # Metadata
    total_images: int
    evaluation_timestamp: str
    created_dataset_path: str
    original_dataset_path: str
    
    def __post_init__(self):
        """Validate data consistency after initialization."""
        if len(self.individual_metrics) != self.total_images:
            raise ValueError(
                f"Mismatch between individual_metrics length "
                f"({len(self.individual_metrics)}) and total_images ({self.total_images})"
            )
    
    @classmethod
    def create_with_timestamp(
        cls,
        individual_metrics: List[IndividualImageMetrics],
        fid_score: float,
        statistics: Dict[str, Dict[str, float]],
        created_dataset_path: str,
        original_dataset_path: str
    ) -> 'AllMetricsResults':
        """Create AllMetricsResults with automatic timestamp.
        
        Args:
            individual_metrics: List of individual image metrics
            fid_score: FID score for the dataset
            statistics: Statistical summaries
            created_dataset_path: Path to test images
            original_dataset_path: Path to reference images
            
        Returns:
            AllMetricsResults: New instance with current timestamp
        """
        return cls(
            individual_metrics=individual_metrics,
            fid_score=fid_score,
            statistics=statistics,
            total_images=len(individual_metrics),
            evaluation_timestamp=datetime.now().isoformat(),
            created_dataset_path=created_dataset_path,
            original_dataset_path=original_dataset_path
        )
    
    def get_metric_summary(self) -> str:
        """Get a summary of the main metrics.
        
        Returns:
            str: Formatted summary string with key metrics
        """
        psnr_mean = self.statistics.get('psnr_db', {}).get('mean', 0)
        ssim_mean = self.statistics.get('ssim', {}).get('mean', 0)
        
        return (
            f"Dataset Metrics Summary:\n"
            f"  Images: {self.total_images}\n"
            f"  PSNR: {psnr_mean:.2f} dB (avg)\n"
            f"  SSIM: {ssim_mean:.4f} (avg)\n"
            f"  FID:  {self.fid_score:.2f}"
        )
    
    def get_detailed_summary(self) -> str:
        """Get a detailed summary including all available metrics.
        
        Returns:
            str: Detailed formatted summary with all metrics and statistics
        """
        lines = [f"Comprehensive Evaluation Results ({self.total_images} images)"]
        lines.append(f"Evaluated: {self.evaluation_timestamp}")
        lines.append(f"Test Dataset: {self.created_dataset_path}")
        lines.append(f"Reference Dataset: {self.original_dataset_path}")
        lines.append("")
        lines.append("Metric Statistics:")
        
        for metric_name, stats in self.statistics.items():
            lines.append(f"  {metric_name.upper()}:")
            for stat_name, value in stats.items():
                if isinstance(value, float):
                    lines.append(f"    {stat_name}: {value:.4f}")
                else:
                    lines.append(f"    {stat_name}: {value}")
        
        lines.append(f"\nDataset-level FID Score: {self.fid_score:.4f}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary format.
        
        Returns:
            Dict[str, Any]: Complete dictionary representation
        """
        return {
            'individual_metrics': [m.to_dict() for m in self.individual_metrics],
            'fid_score': self.fid_score,
            'statistics': self.statistics,
            'total_images': self.total_images,
            'evaluation_timestamp': self.evaluation_timestamp,
            'created_dataset_path': self.created_dataset_path,
            'original_dataset_path': self.original_dataset_path
        }
    
    def save_to_json(self, filepath: str) -> None:
        """Save results to a JSON file.
        
        Args:
            filepath: Path where to save the JSON file
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'AllMetricsResults':
        """Load results from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            AllMetricsResults: Loaded results instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert individual metrics back to dataclass instances
        individual_metrics = [
            IndividualImageMetrics(**metric_data)
            for metric_data in data['individual_metrics']
        ]
        
        return cls(
            individual_metrics=individual_metrics,
            fid_score=data['fid_score'],
            statistics=data['statistics'],
            total_images=data['total_images'],
            evaluation_timestamp=data['evaluation_timestamp'],
            created_dataset_path=data['created_dataset_path'],
            original_dataset_path=data['original_dataset_path']
        )