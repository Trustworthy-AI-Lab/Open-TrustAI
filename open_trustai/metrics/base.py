"""Base classes for fairness metrics."""

from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional, Tuple

import torch
from torch import Tensor


class FairnessMetric(ABC):
    """Base class for all fairness metrics.

    All fairness metrics should inherit from this class and implement the compute method.
    This class provides common functionality for fairness evaluation across different
    metrics.

    Attributes:
        _eps: Small constant to avoid division by zero
        _default_threshold: Default threshold for fairness determination
    """

    def __init__(self, eps: float = 1e-8):
        """Initialize the fairness metric.

        Args:
            eps: Small constant to avoid division by zero. Defaults to 1e-8.
        """
        self._eps = eps
        self._default_threshold = 0.1

    def _validate_inputs(
        self, predictions: Tensor, target: Tensor, sensitive: Tensor
    ) -> None:
        """Validate input tensors.

        Args:
            predictions: Model predictions
            target: Ground truth labels
            sensitive: Sensitive attribute values

        Raises:
            ValueError: If inputs have invalid shapes or types
        """
        if (
            not torch.is_tensor(predictions)
            or not torch.is_tensor(target)
            or not torch.is_tensor(sensitive)
        ):
            raise ValueError("All inputs must be PyTorch tensors")

        if (
            predictions.shape[0] != target.shape[0]
            or predictions.shape[0] != sensitive.shape[0]
        ):
            raise ValueError("All inputs must have the same first dimension")

    @abstractmethod
    def compute(
        self, predictions: Tensor, target: Tensor, sensitive: Tensor
    ) -> Dict[str, float]:
        """Compute the fairness metric.

        Args:
            predictions: Model predictions. Shape: (n_samples,) for binary classification
                        or (n_samples, n_classes) for multi-class.
            target: Ground truth labels. Shape: (n_samples,)
            sensitive: Sensitive attribute values. Shape: (n_samples,)

        Returns:
            Dictionary containing the computed metric values
        """
        self._validate_inputs(predictions, target, sensitive)
        pass

    @abstractmethod
    def get_metric_names(self) -> List[str]:
        """Get the names of metrics computed by this class.

        Returns:
            List of metric names
        """
        pass

    def __call__(
        self, predictions: Tensor, target: Tensor, sensitive: Tensor
    ) -> Dict[str, float]:
        """Convenience method to compute metrics.

        Args:
            predictions: Model predictions
            target: Ground truth labels
            sensitive: Sensitive attribute values

        Returns:
            Dictionary containing the computed metric values
        """
        return self.compute(predictions, target, sensitive)
