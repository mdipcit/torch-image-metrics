"""Dataset evaluator for comprehensive image quality assessment.

This module provides the Evaluator class for dataset-level image quality
evaluation, based on the design from Generative-Latent-Optimization's
SimpleAllMetricsEvaluator. It combines individual image metrics with 
dataset-level metrics like FID to provide comprehensive evaluation results.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union
import warnings
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np

from .calculator import Calculator
from .metrics.dataset import FID
from .core.data_structures import AllMetricsResults, IndividualImageMetrics
from .utils.image_matcher import ImageMatcher

logger = logging.getLogger(__name__)


class Evaluator:
    """Comprehensive dataset evaluator for image quality metrics.
    
    This class evaluates image quality across entire datasets by computing
    individual image metrics (PSNR, SSIM, LPIPS, etc.) and dataset-level 
    metrics (FID). It provides statistical summaries and detailed reporting.
    
    The design is based on SimpleAllMetricsEvaluator from the 
    Generative-Latent-Optimization project, providing a straightforward way
    to compute all metrics for a dataset.
    
    Args:
        device: Device for computation ('cuda' or 'cpu')
        use_lpips: Whether to use LPIPS calculation (default: True)
        use_improved_ssim: Whether to use improved SSIM (default: True) 
        use_fid: Whether to compute FID scores (default: True)
        match_strategy: Strategy for matching images ('stem' or 'full_name')
        image_size: Target size for resizing images (default: None - no resize)
        batch_size: Batch size for processing (default: 32)
        
    Example:
        >>> evaluator = Evaluator(device='cuda')
        >>> results = evaluator.evaluate_dataset('/path/to/test', '/path/to/ref')
        >>> print(results.get_metric_summary())
        >>> evaluator.print_summary(results)
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        use_lpips: bool = True,
        use_improved_ssim: bool = True,
        use_fid: bool = True,
        match_strategy: str = 'stem',
        image_size: Optional[int] = None,
        batch_size: int = 32
    ):
        """Initialize the comprehensive dataset evaluator.
        
        Args:
            device: Target device for computation
            use_lpips: Whether to enable LPIPS metric
            use_improved_ssim: Whether to enable improved SSIM
            use_fid: Whether to enable FID computation
            match_strategy: Image matching strategy ('stem' or 'full_name')
            image_size: Optional target size for square resizing
            batch_size: Batch size for processing
        """
        self.device = device
        self.batch_size = batch_size
        self.image_size = image_size
        
        # Initialize metrics calculator
        logger.info(f"Initializing Evaluator on {device}")
        self.calculator = Calculator(
            device=device,
            use_lpips=use_lpips,
            use_improved_ssim=use_improved_ssim,
            use_fid=use_fid
        )
        
        # Initialize FID evaluator if requested
        self.fid_evaluator = None
        if use_fid:
            try:
                self.fid_evaluator = FID(device=device)
                logger.info("FID evaluator initialized")
            except ImportError:
                logger.warning("FID not available. Install with: pip install pytorch-fid")
        
        # Initialize image matcher
        self.image_matcher = ImageMatcher(match_strategy=match_strategy)
        
        # Setup image preprocessing
        self._setup_preprocessing()
        
        # Log configuration
        logger.info(f"Evaluator configuration:")
        logger.info(f"  Device: {device}")
        logger.info(f"  Metrics: {self.calculator.get_available_metrics()}")
        logger.info(f"  Image matching: {match_strategy}")
        logger.info(f"  Batch size: {batch_size}")
    
    def _setup_preprocessing(self) -> None:
        """Setup image preprocessing pipeline."""
        transform_list = []
        
        # Resize if specified
        if self.image_size is not None:
            transform_list.append(transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=transforms.InterpolationMode.LANCZOS
            ))
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        self.transform = transforms.Compose(transform_list)
    
    def _load_image_as_tensor(self, image_path: Path) -> Optional[torch.Tensor]:
        """Load and preprocess a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Optional[torch.Tensor]: Preprocessed tensor or None if failed
        """
        try:
            # Load image with PIL
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            tensor = self.transform(image)
            
            # Ensure [0, 1] range
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            
            # Clamp to valid range
            tensor = torch.clamp(tensor, 0, 1)
            
            return tensor
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None
    
    def _load_image_pairs(
        self, 
        image_pairs: List[Tuple[Path, Path]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Load and preprocess image pairs.
        
        Args:
            image_pairs: List of (original_path, created_path) pairs
            
        Returns:
            List of (original_tensor, created_tensor) pairs
        """
        tensor_pairs = []
        
        logger.info(f"Loading {len(image_pairs)} image pairs...")
        
        for orig_path, created_path in tqdm(image_pairs, desc="Loading images"):
            try:
                orig_tensor = self._load_image_as_tensor(orig_path)
                created_tensor = self._load_image_as_tensor(created_path)
                
                if orig_tensor is not None and created_tensor is not None:
                    # Ensure same size
                    if orig_tensor.shape != created_tensor.shape:
                        # Resize to match
                        target_h = min(orig_tensor.shape[1], created_tensor.shape[1])
                        target_w = min(orig_tensor.shape[2], created_tensor.shape[2])
                        orig_tensor = F.interpolate(
                            orig_tensor.unsqueeze(0), 
                            size=(target_h, target_w),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)
                        created_tensor = F.interpolate(
                            created_tensor.unsqueeze(0),
                            size=(target_h, target_w),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)
                    
                    tensor_pairs.append((orig_tensor, created_tensor))
                else:
                    logger.warning(f"Failed to load pair: {orig_path.name}")
                
            except Exception as e:
                logger.warning(f"Failed to load image pair {orig_path.name}: {e}")
                continue
        
        return tensor_pairs
    
    def _calculate_individual_metrics(
        self, 
        image_pairs: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> List[IndividualImageMetrics]:
        """Calculate individual metrics for all image pairs.
        
        Args:
            image_pairs: List of (original, created) tensor pairs
            
        Returns:
            List of IndividualImageMetrics for each pair
        """
        results = []
        
        logger.info(f"Computing individual metrics for {len(image_pairs)} pairs...")
        
        for original, created in tqdm(image_pairs, desc="Computing metrics"):
            try:
                # Move to device
                original = original.to(self.device)
                created = created.to(self.device)
                
                # Add batch dimension if needed
                if original.dim() == 3:
                    original = original.unsqueeze(0)
                if created.dim() == 3:
                    created = created.unsqueeze(0)
                
                # Calculate all metrics
                with torch.no_grad():
                    metrics = self.calculator.compute_all_metrics(created, original)
                results.append(metrics)
                
            except Exception as e:
                logger.warning(f"Failed to calculate metrics for pair: {e}")
                # Add placeholder metrics
                results.append(IndividualImageMetrics(
                    psnr_db=0.0, ssim=0.0, mse=1.0, mae=1.0
                ))
        
        return results
    
    def _calculate_dataset_fid(
        self, 
        created_path: Path, 
        original_path: Path
    ) -> float:
        """Calculate FID score between datasets.
        
        Args:
            created_path: Path to created dataset
            original_path: Path to original dataset
            
        Returns:
            FID score or inf if calculation fails
        """
        if self.fid_evaluator is None:
            logger.warning("FID evaluator not initialized")
            return float('inf')
        
        try:
            logger.info("Computing FID score...")
            fid_score = self.fid_evaluator.calculate_dataset(
                str(created_path), 
                str(original_path)
            )
            logger.info(f"FID Score: {fid_score:.2f}")
            return fid_score
        except Exception as e:
            logger.error(f"FID calculation failed: {e}")
            return float('inf')
    
    def _calculate_statistics(
        self, 
        individual_results: List[IndividualImageMetrics]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate comprehensive statistics for all metrics.
        
        Args:
            individual_results: List of IndividualImageMetrics
            
        Returns:
            Dictionary with statistics for each metric
        """
        if not individual_results:
            return {}
        
        statistics = {}
        
        # Basic metrics (always available)
        for metric_name in ['psnr_db', 'ssim', 'mse', 'mae']:
            values = [getattr(r, metric_name) for r in individual_results]
            if values:
                statistics[metric_name] = self._compute_metric_stats(values, metric_name)
        
        # Optional metrics
        lpips_values = [r.lpips for r in individual_results if r.lpips is not None]
        if lpips_values:
            statistics['lpips'] = self._compute_metric_stats(lpips_values, 'LPIPS')
        
        ssim_improved_values = [r.ssim_improved for r in individual_results if r.ssim_improved is not None]
        if ssim_improved_values:
            statistics['ssim_improved'] = self._compute_metric_stats(ssim_improved_values, 'SSIM++')
        
        return statistics
    
    def _compute_metric_stats(self, values: List[float], metric_name: str) -> Dict[str, float]:
        """Compute statistical summary for a single metric.
        
        Args:
            values: List of metric values
            metric_name: Name of the metric for logging
            
        Returns:
            Dictionary with mean, std, min, max, median
        """
        values_array = np.array(values)
        
        stats = {
            'mean': float(np.mean(values_array)),
            'std': float(np.std(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'median': float(np.median(values_array))
        }
        
        logger.info(
            f"  {metric_name}: μ={stats['mean']:.4f}, σ={stats['std']:.4f}, "
            f"range=[{stats['min']:.4f}, {stats['max']:.4f}]"
        )
        
        return stats
    
    def evaluate_dataset(
        self,
        created_dataset_path: Union[str, Path],
        original_dataset_path: Union[str, Path]
    ) -> AllMetricsResults:
        """Evaluate all metrics for a dataset comparison.
        
        This is the main entry point for dataset evaluation, following the
        design of SimpleAllMetricsEvaluator.evaluate_dataset_all_metrics().
        
        Args:
            created_dataset_path: Path to created/test images
            original_dataset_path: Path to original/reference images
            
        Returns:
            AllMetricsResults: Comprehensive evaluation results
            
        Example:
            >>> evaluator = Evaluator(device='cuda')
            >>> results = evaluator.evaluate_dataset('/path/to/test', '/path/to/ref')
            >>> print(f"Evaluated {results.total_images} images")
            >>> print(f"Average PSNR: {results.statistics['psnr_db']['mean']:.2f} dB")
            >>> print(f"FID Score: {results.fid_score:.2f}")
        """
        created_path = Path(created_dataset_path).resolve()
        original_path = Path(original_dataset_path).resolve()
        
        logger.info("=" * 60)
        logger.info("Starting All Metrics Evaluation")
        logger.info("=" * 60)
        logger.info(f"Created dataset:  {created_path}")
        logger.info(f"Original dataset: {original_path}")
        
        # Step 1: Validate and match image pairs
        is_valid, message = self.image_matcher.validate_datasets(created_path, original_path)
        if not is_valid:
            raise ValueError(f"Dataset validation failed: {message}")
        logger.info(message)
        
        # Step 2: Find image pairs
        image_pairs = self.image_matcher.find_image_pairs(created_path, original_path)
        logger.info(f"Found {len(image_pairs)} matching image pairs")
        
        # Step 3: Load image pairs
        tensor_pairs = self._load_image_pairs(image_pairs)
        logger.info(f"Successfully loaded {len(tensor_pairs)} image pairs")
        
        # Step 4: Calculate individual metrics
        individual_results = self._calculate_individual_metrics(tensor_pairs)
        
        # Step 5: Calculate FID score
        fid_score = self._calculate_dataset_fid(created_path, original_path)
        
        # Step 6: Calculate statistics
        logger.info("Computing statistics...")
        statistics = self._calculate_statistics(individual_results)
        
        # Step 7: Create results object
        all_results = AllMetricsResults(
            individual_metrics=individual_results,
            fid_score=fid_score,
            statistics=statistics,
            total_images=len(individual_results),
            evaluation_timestamp=datetime.now().isoformat(),
            created_dataset_path=str(created_path),
            original_dataset_path=str(original_path)
        )
        
        logger.info("=" * 60)
        logger.info("Evaluation Completed Successfully")
        logger.info("=" * 60)
        
        return all_results
    
    def print_summary(self, results: AllMetricsResults) -> None:
        """Print a user-friendly summary of evaluation results.
        
        This provides a formatted output similar to the original 
        SimpleAllMetricsEvaluator.print_summary() method.
        
        Args:
            results: Complete evaluation results
        """
        print("\n" + "=" * 70)
        print("📊 ALL METRICS EVALUATION SUMMARY")
        print("=" * 70)
        
        # Basic info
        print(f"\n📁 Dataset Information:")
        print(f"   Created:  {Path(results.created_dataset_path).name}")
        print(f"   Original: {Path(results.original_dataset_path).name}")
        print(f"   Images:   {results.total_images}")
        print(f"   Time:     {results.evaluation_timestamp}")
        
        # Individual metrics summary
        print(f"\n📈 Individual Metrics Summary:")
        print(f"   {'Metric':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print(f"   {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        
        stats = results.statistics
        for metric in ['psnr_db', 'ssim', 'lpips', 'ssim_improved', 'mse', 'mae']:
            if metric in stats:
                s = stats[metric]
                # Format based on metric type
                if metric == 'psnr_db':
                    print(f"   {'PSNR (dB)':<12} {s['mean']:>10.2f} {s['std']:>10.2f} {s['min']:>10.2f} {s['max']:>10.2f}")
                elif metric in ['ssim', 'ssim_improved', 'lpips']:
                    name = 'SSIM' if metric == 'ssim' else 'SSIM++' if metric == 'ssim_improved' else 'LPIPS'
                    print(f"   {name:<12} {s['mean']:>10.4f} {s['std']:>10.4f} {s['min']:>10.4f} {s['max']:>10.4f}")
                else:
                    name = metric.upper()
                    print(f"   {name:<12} {s['mean']:>10.6f} {s['std']:>10.6f} {s['min']:>10.6f} {s['max']:>10.6f}")
        
        # FID score
        print(f"\n🎯 Dataset-level Metrics:")
        print(f"   FID Score: {results.fid_score:.2f}")
        
        # Quality interpretation
        print(f"\n🏆 Quality Assessment:")
        psnr_mean = stats.get('psnr_db', {}).get('mean', 0)
        ssim_mean = stats.get('ssim', {}).get('mean', 0)
        
        if psnr_mean > 30:
            psnr_quality = "Excellent"
        elif psnr_mean > 25:
            psnr_quality = "Good"
        elif psnr_mean > 20:
            psnr_quality = "Fair"
        else:
            psnr_quality = "Poor"
        
        if ssim_mean > 0.9:
            ssim_quality = "Excellent"
        elif ssim_mean > 0.8:
            ssim_quality = "Good"
        elif ssim_mean > 0.7:
            ssim_quality = "Fair"
        else:
            ssim_quality = "Poor"
        
        if results.fid_score < 20:
            fid_quality = "Excellent"
        elif results.fid_score < 50:
            fid_quality = "Good"
        elif results.fid_score < 100:
            fid_quality = "Fair"
        else:
            fid_quality = "Poor"
        
        print(f"   PSNR Quality:  {psnr_quality} ({psnr_mean:.2f} dB)")
        print(f"   SSIM Quality:  {ssim_quality} ({ssim_mean:.4f})")
        print(f"   FID Quality:   {fid_quality} ({results.fid_score:.2f})")
        
        print("\n" + "=" * 70)
    
    def evaluate_image_pair(
        self,
        created_image: Union[str, Path],
        original_image: Union[str, Path]
    ) -> IndividualImageMetrics:
        """Evaluate a single image pair.
        
        Args:
            created_image: Path to created/test image
            original_image: Path to original/reference image
            
        Returns:
            IndividualImageMetrics: Metrics for the image pair
        """
        created_path = Path(created_image)
        original_path = Path(original_image)
        
        # Load images
        created_tensor = self._load_image_as_tensor(created_path)
        original_tensor = self._load_image_as_tensor(original_path)
        
        if created_tensor is None or original_tensor is None:
            raise ValueError("Failed to load one or both images")
        
        # Calculate metrics
        tensor_pairs = [(original_tensor, created_tensor)]
        results = self._calculate_individual_metrics(tensor_pairs)
        
        return results[0] if results else None