"""Tests for Evaluator class.

This module tests the comprehensive Evaluator class that handles dataset-level
image quality evaluation, based on the design from SimpleAllMetricsEvaluator.
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from PIL import Image

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root / "src"))

from torch_image_metrics.evaluator import Evaluator
from torch_image_metrics.utils.image_matcher import ImageMatcher


def create_test_image(size=(64, 64), color=(255, 0, 0)):
    """Create a test image with specified size and color."""
    image = Image.new('RGB', size, color)
    return image


@pytest.fixture
def temp_dataset_dir():
    """Create temporary dataset directories with test images."""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    # Create test and reference directories
    test_dir = temp_path / "test"
    ref_dir = temp_path / "ref"
    test_dir.mkdir()
    ref_dir.mkdir()
    
    # Create some test images
    for i in range(3):
        # Test images (slightly different)
        test_image = create_test_image(color=(200, 100, 50))
        test_image.save(test_dir / f"image_{i:03d}.png")
        
        # Reference images 
        ref_image = create_test_image(color=(255, 128, 64))
        ref_image.save(ref_dir / f"image_{i:03d}.png")
    
    yield test_dir, ref_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_evaluator():
    """Create a sample evaluator for testing."""
    return Evaluator(
        device='cpu',
        use_lpips=False,  # Disable for faster tests
        use_fid=False,    # Disable for faster tests
        batch_size=2
    )


class TestEvaluatorInitialization:
    """Test evaluator initialization."""
    
    def test_default_initialization(self):
        """Test evaluator with default parameters."""
        evaluator = Evaluator(device='cpu')
        
        assert evaluator.device == 'cpu'
        assert evaluator.image_matcher is not None
        assert evaluator.calculator is not None
    
    def test_custom_parameters(self):
        """Test evaluator with custom parameters."""
        evaluator = Evaluator(
            device='cpu',
            use_lpips=False,
            use_improved_ssim=False,
            use_fid=False,
            match_strategy='full_name',
            image_size=128,
            batch_size=16
        )
        
        assert evaluator.device == 'cpu'
        assert evaluator.batch_size == 16
        assert evaluator.image_size == 128
        assert evaluator.image_matcher.match_strategy == 'full_name'


class TestImageProcessing:
    """Test image loading and processing functionality."""
    
    def test_load_image_as_tensor(self, temp_dataset_dir, sample_evaluator):
        """Test loading a single image as tensor."""
        test_dir, ref_dir = temp_dataset_dir
        image_path = list(test_dir.glob("*.png"))[0]
        
        tensor = sample_evaluator._load_image_as_tensor(image_path)
        
        assert tensor is not None
        assert tensor.dim() == 3  # C, H, W
        assert tensor.shape[0] == 3  # RGB channels
        assert 0 <= tensor.min() <= tensor.max() <= 1  # [0, 1] range
    
    def test_load_nonexistent_image(self, sample_evaluator):
        """Test loading nonexistent image returns None."""
        fake_path = Path("/fake/path/image.png")
        tensor = sample_evaluator._load_image_as_tensor(fake_path)
        assert tensor is None
    
    def test_load_image_pairs(self, temp_dataset_dir, sample_evaluator):
        """Test loading multiple image pairs."""
        test_dir, ref_dir = temp_dataset_dir
        
        # Get image paths
        test_files = list(test_dir.glob("*.png"))
        ref_files = list(ref_dir.glob("*.png"))
        
        # Create pairs
        image_pairs = [(ref_files[i], test_files[i]) for i in range(len(test_files))]
        
        tensor_pairs = sample_evaluator._load_image_pairs(image_pairs)
        
        assert len(tensor_pairs) == len(image_pairs)
        for orig_tensor, created_tensor in tensor_pairs:
            assert orig_tensor.dim() == 3
            assert created_tensor.dim() == 3
            assert orig_tensor.shape == created_tensor.shape


class TestMetricsCalculation:
    """Test metrics calculation functionality."""
    
    def test_calculate_individual_metrics(self, sample_evaluator):
        """Test individual metrics calculation."""
        # Create sample tensor pairs
        tensor_pairs = [
            (torch.rand(3, 32, 32), torch.rand(3, 32, 32)),
            (torch.rand(3, 32, 32), torch.rand(3, 32, 32))
        ]
        
        results = sample_evaluator._calculate_individual_metrics(tensor_pairs)
        
        assert len(results) == 2
        for result in results:
            assert hasattr(result, 'psnr_db')
            assert hasattr(result, 'ssim')
            assert hasattr(result, 'mse')
            assert hasattr(result, 'mae')
            assert isinstance(result.psnr_db, float)
            assert isinstance(result.ssim, float)
            assert result.psnr_db > 0
            assert 0 <= result.ssim <= 1
    
    def test_calculate_statistics(self, sample_evaluator):
        """Test statistics calculation."""
        from torch_image_metrics.core.data_structures import IndividualImageMetrics
        
        # Create sample results
        results = [
            IndividualImageMetrics(psnr_db=25.0, ssim=0.8, mse=0.01, mae=0.05),
            IndividualImageMetrics(psnr_db=30.0, ssim=0.9, mse=0.005, mae=0.03),
            IndividualImageMetrics(psnr_db=28.0, ssim=0.85, mse=0.008, mae=0.04)
        ]
        
        statistics = sample_evaluator._calculate_statistics(results)
        
        # Check that statistics are computed for all basic metrics
        for metric in ['psnr_db', 'ssim', 'mse', 'mae']:
            assert metric in statistics
            stats = statistics[metric]
            assert 'mean' in stats
            assert 'std' in stats
            assert 'min' in stats
            assert 'max' in stats
            assert 'median' in stats
        
        # Verify some basic properties
        assert statistics['psnr_db']['mean'] == pytest.approx(27.67, abs=0.1)
        assert statistics['ssim']['mean'] == pytest.approx(0.85, abs=0.01)


class TestEvaluatorMainFunctionality:
    """Test main evaluator functionality."""
    
    def test_evaluate_image_pair(self, temp_dataset_dir, sample_evaluator):
        """Test evaluating a single image pair."""
        test_dir, ref_dir = temp_dataset_dir
        
        test_image = list(test_dir.glob("*.png"))[0]
        ref_image = list(ref_dir.glob("*.png"))[0]
        
        result = sample_evaluator.evaluate_image_pair(test_image, ref_image)
        
        assert result is not None
        assert hasattr(result, 'psnr_db')
        assert hasattr(result, 'ssim') 
        assert result.psnr_db > 0
        assert 0 <= result.ssim <= 1
    
    def test_evaluate_dataset_minimal(self, temp_dataset_dir):
        """Test basic dataset evaluation with minimal features."""
        test_dir, ref_dir = temp_dataset_dir
        
        # Use minimal evaluator (no LPIPS, no FID)
        evaluator = Evaluator(
            device='cpu',
            use_lpips=False,
            use_improved_ssim=False,
            use_fid=False,
            batch_size=1
        )
        
        results = evaluator.evaluate_dataset(test_dir, ref_dir)
        
        # Check basic results structure
        assert results is not None
        assert results.total_images == 3  # We created 3 pairs
        assert len(results.individual_metrics) == 3
        assert isinstance(results.fid_score, float)  # Should be inf due to no FID
        assert len(results.statistics) > 0
        
        # Check individual metrics
        for metric in results.individual_metrics:
            assert metric.psnr_db > 0
            assert 0 <= metric.ssim <= 1
            assert metric.mse >= 0
            assert metric.mae >= 0
    
    def test_print_summary(self, temp_dataset_dir, sample_evaluator, capsys):
        """Test printing evaluation summary."""
        test_dir, ref_dir = temp_dataset_dir
        
        results = sample_evaluator.evaluate_dataset(test_dir, ref_dir)
        sample_evaluator.print_summary(results)
        
        captured = capsys.readouterr()
        assert "ALL METRICS EVALUATION SUMMARY" in captured.out
        assert "PSNR" in captured.out
        assert "SSIM" in captured.out
        assert "FID Score" in captured.out


class TestEvaluatorEdgeCases:
    """Test edge cases and error handling."""
    
    def test_nonexistent_directories(self, sample_evaluator):
        """Test evaluation with nonexistent directories."""
        with pytest.raises(ValueError, match="Dataset validation failed"):
            sample_evaluator.evaluate_dataset("/fake/test", "/fake/ref")
    
    def test_empty_directories(self, sample_evaluator):
        """Test evaluation with empty directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "test"
            ref_dir = Path(temp_dir) / "ref"
            test_dir.mkdir()
            ref_dir.mkdir()
            
            with pytest.raises(ValueError, match="validation failed"):
                sample_evaluator.evaluate_dataset(test_dir, ref_dir)
    
    def test_mismatched_images(self, sample_evaluator):
        """Test evaluation with mismatched image sets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_dir = temp_path / "test"
            ref_dir = temp_path / "ref"
            test_dir.mkdir()
            ref_dir.mkdir()
            
            # Create mismatched sets
            create_test_image().save(test_dir / "image1.png")
            create_test_image().save(ref_dir / "different_name.png")
            
            with pytest.raises(ValueError, match="validation failed"):
                sample_evaluator.evaluate_dataset(test_dir, ref_dir)


class TestImageMatcher:
    """Test ImageMatcher utility class."""
    
    def test_image_matcher_initialization(self):
        """Test ImageMatcher initialization."""
        matcher = ImageMatcher(match_strategy='stem')
        assert matcher.match_strategy == 'stem'
        
        matcher = ImageMatcher(match_strategy='full_name')
        assert matcher.match_strategy == 'full_name'
    
    def test_invalid_match_strategy(self):
        """Test invalid match strategy raises error."""
        with pytest.raises(ValueError, match="Invalid match strategy"):
            ImageMatcher(match_strategy='invalid')
    
    def test_get_image_files(self, temp_dataset_dir):
        """Test getting image files from directory."""
        test_dir, ref_dir = temp_dataset_dir
        matcher = ImageMatcher()
        
        files = matcher.get_image_files(test_dir)
        assert len(files) == 3
        assert all(f.suffix.lower() == '.png' for f in files)
    
    def test_match_by_stem(self, temp_dataset_dir):
        """Test matching images by stem."""
        test_dir, ref_dir = temp_dataset_dir
        matcher = ImageMatcher(match_strategy='stem')
        
        pairs = matcher.find_image_pairs(test_dir, ref_dir)
        assert len(pairs) == 3
        
        for orig_path, created_path in pairs:
            assert orig_path.stem == created_path.stem
    
    def test_validate_datasets(self, temp_dataset_dir):
        """Test dataset validation."""
        test_dir, ref_dir = temp_dataset_dir
        matcher = ImageMatcher()
        
        is_valid, message = matcher.validate_datasets(test_dir, ref_dir)
        assert is_valid is True
        assert "validation successful" in message.lower()
    
    def test_matching_statistics(self, temp_dataset_dir):
        """Test getting matching statistics."""
        test_dir, ref_dir = temp_dataset_dir
        matcher = ImageMatcher()
        
        stats = matcher.get_matching_statistics(test_dir, ref_dir)
        
        assert stats['total_original'] == 3
        assert stats['total_created'] == 3
        assert stats['matched_pairs'] == 3
        assert stats['unmatched_original'] == 0
        assert stats['unmatched_created'] == 0


if __name__ == '__main__':
    pytest.main([__file__])