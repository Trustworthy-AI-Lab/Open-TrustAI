"""Base dataset classes for Open-TrustAI."""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from PIL import Image


class FairDataset(Dataset, ABC):
    """Base class for all fairness-aware datasets.

    This abstract class defines the interface that all datasets must implement
    and provides common functionality for data loading and preprocessing.
    """

    def __init__(
        self,
        root: str,
        download: bool = False,
        target_attribute: Optional[str] = None,
        sensitive_attribute: Optional[str] = None,
        feature_transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        sensitive_transform: Optional[Any] = None,
    ):
        """Initialize the dataset.

        Args:
            root: Root directory for the dataset
            download: Whether to download the dataset if not present
            target_attribute: Target attribute choosed to use (if multiple available)
            sensitive_attribute: Sensitive attribute choosed to use (if multiple available)
            feature_transform: Transform to apply to features
            target_transform: Transform to apply to targets
            sensitive_transform: Transform to apply to sensitive attributes
        """
        self.root = os.path.expanduser(root)
        self.target_attribute = target_attribute
        self.sensitive_attribute = sensitive_attribute
        self.feature_transform = feature_transform
        self.target_transform = target_transform
        self.sensitive_transform = sensitive_transform

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. Use download=True to download it.")

        self._load_data()

    @abstractmethod
    def _check_exists(self) -> bool:
        """Check if the dataset exists in the root directory."""
        pass

    @abstractmethod
    def _download(self) -> None:
        """Download the dataset if it doesn't exist."""
        pass

    @abstractmethod
    def _load_data(self) -> None:
        """Load the dataset into memory."""
        pass

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """Get the list of feature names.

        Returns:
            List of feature names
        """
        pass

    @property
    @abstractmethod
    def sensitive_attribute_names(self) -> List[str]:
        """Get the list of [possible] sensitive attribute names.
        The choosed one is specified in the constructor and must be in the list.


        Returns:
            List of sensitive attribute names
        """
        pass

    @property
    @abstractmethod
    def protected_groups(self) -> Dict[str, List[Any]]:
        """Get the protected groups for each sensitive attribute.

        Returns:
            Dictionary mapping sensitive attributes to their protected group values
            For example: {"sex": ["Female"], "race": ["Black", "Asian", "Hispanic"], "age": [">=40"]}
        """
        pass

    @property
    @abstractmethod
    def target_attribute_names(self) -> List[str]:
        """Get the list of [possible] target attribute names.
        The choosed one is specified in the constructor and must be in the list.

        Returns:
            List of target attribute names
        """
        pass

    @property
    @abstractmethod
    def favourable_outcomes(self) -> Dict[str, List[Any]]:
        """Get the favourable outcome values for each target attribute.

        Returns:
            Dictionary mapping target attribute names to lists of their favorable outcome values.
            For example: {"income": [">50K"], "loan": ["approved"], "recidivism": ["no_recid"]}
        """
        pass

    @property
    @abstractmethod
    def recommended_metrics(self) -> List[str]:
        """Get the recommended fairness metrics for this dataset.


        Returns:
            List of recommended metric names
        """
        pass

    @property
    @abstractmethod
    def bias_notes(self) -> str:
        """Get notes about known biases in the dataset.

        Returns:
            String containing bias notes
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.

        Args:
            index: Index of the sample to get

        Returns:
            A tuple containing the features, target, and sensitive attributes
        """
        pass

    def get_fairness_info(self) -> Dict[str, Any]:
        """Get information about fairness-related aspects of the dataset.


        Returns:
            Dictionary containing:
                - sensitive_attributes: List of sensitive attribute names
                - target_attributes: List of target attribute names
                - protected_groups: Dictionary mapping sensitive attributes to protected groups
                - fairness_metrics: List of recommended fairness metrics
                - bias_notes: Notes about known biases in the dataset
        """
        return {
            "sensitive_attributes": self.sensitive_attribute_names,
            "target_attributes": self.target_attribute_names,
            "protected_groups": self.protected_groups,
            "fairness_metrics": self.recommended_metrics,
            "bias_notes": self.bias_notes,
        }


class TabularDataset(FairDataset):
    """Base class for tabular datasets.

    This class provides common functionality for handling tabular data,
    including data preprocessing, encoding, and scaling.
    """

    VALID_ENCODINGS = ["label", "onehot", "none"]
    VALID_SCALINGS = ["standard", "minmax", "none"]

    def __init__(
        self,
        root: str,
        download: bool = False,
        target_attribute: Optional[str] = None,
        sensitive_attribute: Optional[str] = None,
        feature_transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        sensitive_transform: Optional[Any] = None,
        categorical_encoding: str = "label",
        numerical_scaling: str = "standard",
    ):
        """Initialize the tabular dataset.

        Args:
            root: Root directory for the dataset
            download: Whether to download the dataset if not present
            target_attribute: Target attribute to use (if multiple available)
            sensitive_attribute: Sensitive attribute to use (if multiple available)
            feature_transform: Transform to apply to features
            target_transform: Transform to apply to targets
            sensitive_transform: Transform to apply to sensitive attributes
            categorical_encoding: Method for encoding categorical variables
                ("label", "onehot", or "none")
            numerical_scaling: Method for scaling numerical variables
                ("standard", "minmax", or "none")

        Raises:
            ValueError: If invalid encoding or scaling method is specified
        """
        if categorical_encoding not in self.VALID_ENCODINGS:
            raise ValueError(
                f"Invalid categorical encoding method. Must be one of {self.VALID_ENCODINGS}"
            )
        if numerical_scaling not in self.VALID_SCALINGS:
            raise ValueError(
                f"Invalid numerical scaling method. Must be one of {self.VALID_SCALINGS}"
            )

        self.categorical_encoding = categorical_encoding
        self.numerical_scaling = numerical_scaling

        super().__init__(
            root=root,
            download=download,
            target_attribute=target_attribute,
            sensitive_attribute=sensitive_attribute,
            feature_transform=feature_transform,
            target_transform=target_transform,
            sensitive_transform=sensitive_transform,
        )

    @property
    @abstractmethod
    def categorical_features(self) -> List[str]:
        """Get the list of categorical feature names.

        Returns:
            List of categorical feature names
        """
        pass

    @property
    @abstractmethod
    def numerical_features(self) -> List[str]:
        """Get the list of numerical feature names.

        Returns:
            List of numerical feature names
        """
        pass

    def _preprocess_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess categorical features.

        Args:
            data: Input DataFrame

        Returns:
            Preprocessed DataFrame
        """
        if self.categorical_encoding == "none":
            return data

        result = data.copy()
        for col in self.categorical_features:
            if col in result.columns:
                if self.categorical_encoding == "label":
                    result[col] = pd.Categorical(result[col]).codes
                elif self.categorical_encoding == "onehot":
                    dummies = pd.get_dummies(result[col], prefix=col)
                    result = pd.concat([result, dummies], axis=1)
                    result.drop(col, axis=1, inplace=True)

        return result

    def _preprocess_numerical(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess numerical features.

        Args:
            data: Input DataFrame

        Returns:
            Preprocessed DataFrame

        Raises:
            ValueError: If numerical feature has zero variance
        """
        if self.numerical_scaling == "none":
            return data

        result = data.copy()
        for col in self.numerical_features:
            if col in result.columns:
                if self.numerical_scaling == "standard":
                    std = result[col].std()
                    if std == 0:
                        raise ValueError(f"Feature {col} has zero variance")
                    result[col] = (result[col] - result[col].mean()) / std
                elif self.numerical_scaling == "minmax":
                    min_val = result[col].min()
                    max_val = result[col].max()
                    if min_val == max_val:
                        raise ValueError(f"Feature {col} has zero range")
                    result[col] = (result[col] - min_val) / (max_val - min_val)

        return result

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample to get

        Returns:
            Tuple of (feature, target, sensitive) arrays
        """
        return self.features[idx], self.target[idx], self.sensitive[idx]


class VisionDataset(FairDataset):
    """Base class for vision datasets.

    This class provides common functionality for handling image data,
    including data loading, transformations, and caching.
    """

    def __init__(
        self,
        root: str,
        download: bool = False,
        target_attribute: Optional[str] = None,
        sensitive_attribute: Optional[str] = None,
        feature_transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        sensitive_transform: Optional[Any] = None,
        image_size: Tuple[int, int] = (224, 224),
        cache_images: bool = False,
    ):
        """Initialize the vision dataset.

        Args:
            root: Root directory for the dataset
            download: Whether to download the dataset if not present
            target_attribute: Target attribute to use (if multiple available)
            sensitive_attribute: Sensitive attribute to use (if multiple available)
            feature_transform: Transform to apply to features (images)
            target_transform: Transform to apply to targets
            sensitive_transform: Transform to apply to sensitive attributes
            image_size: Size to resize images to (height, width)
            cache_images: Whether to cache images in memory for faster access
        """
        self.image_size = image_size
        self.cache_images = cache_images
        self._image_cache = {}
        self.data = None  # Will store image paths and metadata

        if feature_transform is None:
            feature_transform = self._get_default_transforms()

        super().__init__(
            root=root,
            download=download,
            target_attribute=target_attribute,
            sensitive_attribute=sensitive_attribute,
            feature_transform=feature_transform,
            target_transform=target_transform,
            sensitive_transform=sensitive_transform,
        )
