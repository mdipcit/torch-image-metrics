"""Core components for torch-image-metrics

This module contains the base classes and data structures that form
the foundation of the metrics calculation system.
"""

# Import core classes and data structures
from .base_metric import BaseMetric
from .data_structures import IndividualImageMetrics, AllMetricsResults

__all__ = [
    "BaseMetric",
    "IndividualImageMetrics",
    "AllMetricsResults",
]