"""Image matching utilities for dataset evaluation.

This module provides the ImageMatcher class for finding corresponding images
between two datasets based on filename patterns. This is essential for 
dataset-level evaluation where images need to be paired for comparison.
"""

from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ImageMatcher:
    """Handles matching of image pairs between datasets by filename.
    
    This class provides flexible strategies for matching images between
    test and reference datasets, which is crucial for evaluation pipelines.
    
    Args:
        match_strategy: Strategy for matching images ('stem' or 'full_name')
            - 'stem': Match by filename without extension (default)
            - 'full_name': Match by complete filename including extension
    
    Example:
        >>> matcher = ImageMatcher(match_strategy='stem')
        >>> pairs = matcher.find_image_pairs('/path/to/test', '/path/to/ref')
        >>> print(f"Found {len(pairs)} matching image pairs")
    """
    
    def __init__(self, match_strategy: str = 'stem'):
        """Initialize image matcher.
        
        Args:
            match_strategy: Strategy for matching images ('stem' or 'full_name')
        """
        if match_strategy not in ['stem', 'full_name']:
            raise ValueError(f"Invalid match strategy: {match_strategy}. Use 'stem' or 'full_name'")
        self.match_strategy = match_strategy
        
    def get_image_files(self, path: Union[str, Path]) -> List[Path]:
        """Get all image files from a directory or single file.
        
        Args:
            path: Directory path or single file path
            
        Returns:
            List of image file Path objects
            
        Raises:
            ValueError: If path is invalid or not an image file
        """
        path = Path(path)
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
        image_files = []
        
        # Handle single file
        if path.is_file():
            if path.suffix.lower() in image_extensions:
                return [path]
            else:
                raise ValueError(f"File {path} is not a supported image format")
        
        # Handle directory
        if not path.is_dir():
            raise ValueError(f"Path {path} is not a valid file or directory")
        
        # Collect all image files
        for ext in image_extensions:
            image_files.extend(path.glob(f"*{ext}"))
            image_files.extend(path.glob(f"*{ext.upper()}"))
            # Also check subdirectories (one level deep)
            image_files.extend(path.glob(f"*/*{ext}"))
            image_files.extend(path.glob(f"*/*{ext.upper()}"))
        
        # Remove duplicates and sort
        image_files = list(set(image_files))
        return sorted(image_files)
    
    def match_image_pairs(
        self, 
        original_images: List[Path], 
        created_images: List[Path]
    ) -> List[Tuple[Path, Path]]:
        """Match images between original and created datasets.
        
        Args:
            original_images: List of original/reference image paths
            created_images: List of created/test image paths
            
        Returns:
            List of (original_path, created_path) pairs
        """
        if self.match_strategy == 'stem':
            return self._match_by_stem(original_images, created_images)
        elif self.match_strategy == 'full_name':
            return self._match_by_full_name(original_images, created_images)
        else:
            raise ValueError(f"Unknown match strategy: {self.match_strategy}")
    
    def _match_by_stem(
        self, 
        original_images: List[Path], 
        created_images: List[Path]
    ) -> List[Tuple[Path, Path]]:
        """Match images by filename stem (without extension).
        
        This is the most common matching strategy, useful when test and
        reference images have the same base names but different extensions.
        
        Args:
            original_images: List of original image paths
            created_images: List of created image paths
            
        Returns:
            List of (original_path, created_path) pairs
        """
        # Create dictionaries for efficient lookup
        original_dict = {img.stem: img for img in original_images}
        created_dict = {img.stem: img for img in created_images}
        
        # Find matching pairs
        pairs = []
        for stem in sorted(original_dict.keys()):
            if stem in created_dict:
                pairs.append((original_dict[stem], created_dict[stem]))
        
        return pairs
    
    def _match_by_full_name(
        self, 
        original_images: List[Path], 
        created_images: List[Path]
    ) -> List[Tuple[Path, Path]]:
        """Match images by full filename (including extension).
        
        This strategy requires exact filename matches including extensions.
        
        Args:
            original_images: List of original image paths
            created_images: List of created image paths
            
        Returns:
            List of (original_path, created_path) pairs
        """
        # Create dictionaries for efficient lookup
        original_dict = {img.name: img for img in original_images}
        created_dict = {img.name: img for img in created_images}
        
        # Find matching pairs
        pairs = []
        for name in sorted(original_dict.keys()):
            if name in created_dict:
                pairs.append((original_dict[name], created_dict[name]))
        
        return pairs
    
    def find_image_pairs(
        self, 
        created_path: Union[str, Path], 
        original_path: Union[str, Path]
    ) -> List[Tuple[Path, Path]]:
        """Complete workflow to find matching image pairs between two datasets.
        
        This is the main entry point for matching images. It handles directory
        traversal, image discovery, and pairing.
        
        Args:
            created_path: Path to created/test dataset
            original_path: Path to original/reference dataset
            
        Returns:
            List of (original_path, created_path) pairs
            
        Raises:
            ValueError: If no matching pairs are found
        """
        created_path = Path(created_path)
        original_path = Path(original_path)
        
        # Get image files from both datasets
        created_images = self.get_image_files(created_path)
        original_images = self.get_image_files(original_path)
        
        logger.info(f"Found {len(created_images)} created/test images")
        logger.info(f"Found {len(original_images)} original/reference images")
        
        # Match images by filename
        image_pairs = self.match_image_pairs(original_images, created_images)
        
        if not image_pairs:
            raise ValueError(
                f"No matching image pairs found between {created_path} and {original_path}. "
                f"Ensure both directories contain images with matching filenames (strategy: {self.match_strategy})"
            )
        
        logger.info(f"Matched {len(image_pairs)} image pairs using strategy '{self.match_strategy}'")
        
        return image_pairs
    
    def get_matching_statistics(
        self, 
        created_path: Union[str, Path], 
        original_path: Union[str, Path]
    ) -> Dict[str, int]:
        """Get statistics about matching between two datasets.
        
        This is useful for understanding the coverage of your evaluation.
        
        Args:
            created_path: Path to created/test dataset
            original_path: Path to original/reference dataset
            
        Returns:
            Dictionary with matching statistics:
                - total_original: Number of original images
                - total_created: Number of created images
                - matched_pairs: Number of successfully matched pairs
                - unmatched_original: Original images without matches
                - unmatched_created: Created images without matches
        """
        created_images = self.get_image_files(Path(created_path))
        original_images = self.get_image_files(Path(original_path))
        image_pairs = self.match_image_pairs(original_images, created_images)
        
        # Find unmatched images
        matched_originals = set(p[0] for p in image_pairs)
        matched_created = set(p[1] for p in image_pairs)
        
        return {
            'total_original': len(original_images),
            'total_created': len(created_images),
            'matched_pairs': len(image_pairs),
            'unmatched_original': len(original_images) - len(matched_originals),
            'unmatched_created': len(created_images) - len(matched_created)
        }
    
    def validate_datasets(
        self, 
        created_path: Union[str, Path], 
        original_path: Union[str, Path]
    ) -> Tuple[bool, str]:
        """Validate that datasets are ready for evaluation.
        
        Args:
            created_path: Path to created/test dataset
            original_path: Path to original/reference dataset
            
        Returns:
            Tuple of (is_valid, message) where is_valid indicates if datasets
            can be evaluated and message provides details
        """
        try:
            stats = self.get_matching_statistics(created_path, original_path)
            
            if stats['matched_pairs'] == 0:
                return False, "No matching image pairs found"
            
            message = (
                f"Dataset validation successful: "
                f"{stats['matched_pairs']} matched pairs, "
                f"{stats['unmatched_original']} unmatched originals, "
                f"{stats['unmatched_created']} unmatched created"
            )
            
            if stats['unmatched_original'] > 0 or stats['unmatched_created'] > 0:
                message += " (Warning: Not all images have matches)"
            
            return True, message
            
        except Exception as e:
            return False, f"Validation failed: {str(e)}"