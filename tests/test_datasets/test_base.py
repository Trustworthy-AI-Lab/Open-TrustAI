"""Tests for the base dataset classes."""

import os
import pytest
import torch
import numpy as np
from typing import Dict, List, Any

from open_trustai.datasets.base import FairDataset, TabularDataset, VisionDataset


class DummyFairDataset(FairDataset):
    """Dummy dataset for testing FairDataset base class."""

    def __init__(
        self,
        root: str = "./data",
        target_attribute: str = "income",
        sensitive_attribute: str = "gender",
    ):
        super().__init__(
            root,
            download=False,
            target_attribute=target_attribute,
            sensitive_attribute=sensitive_attribute,
        )
        # Set seed for reproducible data
        torch.manual_seed(42)
        self.data = torch.randn(100, 10)  # 100 samples, 10 features
        self.targets = torch.randint(0, 2, (100,))  # Binary targets
        self.sensitive = torch.randint(0, 2, (100, 2))  # Two sensitive attributes

    def _check_exists(self) -> bool:
        return True

    def _download(self) -> None:
        pass

    def _load_data(self) -> None:
        pass

    @property
    def feature_names(self) -> List[str]:
        return [f"feature_{i}" for i in range(10)]

    @property
    def sensitive_attribute_names(self) -> List[str]:
        return ["gender", "race"]

    @property
    def target_attribute_names(self) -> List[str]:
        return ["income", "employment"]

    @property
    def protected_groups(self) -> Dict[str, List[Any]]:
        return {"gender": ["Female"], "race": ["Black", "Hispanic"]}

    @property
    def favourable_outcomes(self) -> Dict[str, List[Any]]:
        return {"income": [">50K"], "employment": ["employed"]}

    @property
    def recommended_metrics(self) -> List[str]:
        return ["statistical_parity", "equal_opportunity"]

    @property
    def bias_notes(self) -> str:
        return "Test bias notes"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        data = self.data[idx]
        target = self.targets[idx]
        sensitive = self.sensitive[idx]

        if self.feature_transform is not None:
            data = self.feature_transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.sensitive_transform is not None:
            sensitive = self.sensitive_transform(sensitive)

        return data, target, sensitive


class DummyTabularDataset(TabularDataset):
    """Dummy dataset for testing TabularDataset base class."""

    def __init__(
        self,
        root: str = "./data",
        target_attribute: str = "income",
        sensitive_attribute: str = "gender",
        categorical_encoding: str = "label",
        numerical_scaling: str = "standard",
    ):
        super().__init__(
            root,
            download=False,
            target_attribute=target_attribute,
            sensitive_attribute=sensitive_attribute,
            categorical_encoding=categorical_encoding,
            numerical_scaling=numerical_scaling,
        )
        # Set seed for reproducible data
        torch.manual_seed(42)
        self.data = torch.randn(100, 10)
        self.targets = torch.randint(0, 2, (100,))
        self.sensitive = torch.randint(0, 2, (100, 2))

    @property
    def feature_names(self) -> List[str]:
        return [f"feature_{i}" for i in range(10)]

    @property
    def sensitive_attribute_names(self) -> List[str]:
        return ["gender", "race"]

    @property
    def target_attribute_names(self) -> List[str]:
        return ["income", "employment"]

    @property
    def categorical_features(self) -> List[str]:
        return ["feature_0", "feature_1"]

    @property
    def numerical_features(self) -> List[str]:
        return [f"feature_{i}" for i in range(2, 10)]

    @property
    def protected_groups(self) -> Dict[str, List[Any]]:
        return {"gender": ["Female"], "race": ["Black", "Hispanic"]}

    @property
    def favourable_outcomes(self) -> Dict[str, List[Any]]:
        return {"income": [">50K"], "employment": ["employed"]}

    @property
    def recommended_metrics(self) -> List[str]:
        return ["statistical_parity", "equal_opportunity"]

    @property
    def bias_notes(self) -> str:
        return "Test bias notes"

    def _check_exists(self) -> bool:
        return True

    def _download(self) -> None:
        pass

    def _load_data(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        data = self.data[idx]
        target = self.targets[idx]
        sensitive = self.sensitive[idx]

        if self.feature_transform is not None:
            data = self.feature_transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.sensitive_transform is not None:
            sensitive = self.sensitive_transform(sensitive)

        return data, target, sensitive


@pytest.fixture
def fair_dataset():
    """Fixture for testing FairDataset."""
    return DummyFairDataset()


@pytest.fixture
def tabular_dataset():
    """Fixture for testing TabularDataset."""
    return DummyTabularDataset()


def test_fair_dataset_init(fair_dataset):
    """Test FairDataset initialization."""
    assert len(fair_dataset) == 100
    assert fair_dataset.feature_names == [f"feature_{i}" for i in range(10)]
    assert fair_dataset.sensitive_attribute_names == ["gender", "race"]
    assert fair_dataset.target_attribute_names == ["income", "employment"]


def test_fair_dataset_getitem(fair_dataset):
    """Test FairDataset __getitem__."""
    data, target, sensitive = fair_dataset[0]
    assert data.shape == (10,)
    assert isinstance(target, torch.Tensor)
    assert sensitive.shape == (2,)


def test_fair_dataset_fairness_info(fair_dataset):
    """Test FairDataset fairness information."""
    info = fair_dataset.get_fairness_info()
    assert "sensitive_attributes" in info
    assert "target_attributes" in info
    assert "protected_groups" in info
    assert "fairness_metrics" in info
    assert "bias_notes" in info
    assert info["sensitive_attributes"] == ["gender", "race"]
    assert info["target_attributes"] == ["income", "employment"]
    assert info["protected_groups"] == {
        "gender": ["Female"],
        "race": ["Black", "Hispanic"],
    }
    assert "statistical_parity" in info["fairness_metrics"]


def test_tabular_dataset_init(tabular_dataset):
    """Test TabularDataset initialization."""
    assert len(tabular_dataset) == 100
    assert len(tabular_dataset.categorical_features) == 2
    assert len(tabular_dataset.numerical_features) == 8


def test_tabular_dataset_invalid_params():
    """Test TabularDataset with invalid parameters."""
    with pytest.raises(ValueError, match="Invalid categorical encoding"):
        DummyTabularDataset(categorical_encoding="invalid")
    with pytest.raises(ValueError, match="Invalid numerical scaling"):
        DummyTabularDataset(numerical_scaling="invalid")


def test_tabular_dataset_preprocessing(tabular_dataset):
    """Test TabularDataset preprocessing methods."""
    import pandas as pd

    # Create test data
    data = pd.DataFrame(
        {
            "feature_0": ["A", "B", "A", "C"],
            "feature_1": ["X", "Y", "X", "Z"],
            "feature_2": [1.0, 2.0, 3.0, 4.0],
            "feature_3": [0.1, 0.2, 0.3, 0.4],
        }
    )

    # Test categorical preprocessing
    cat_data = tabular_dataset._preprocess_categorical(data)
    assert cat_data["feature_0"].dtype in [np.int8, np.int16, np.int32, np.int64]
    assert cat_data["feature_1"].dtype in [np.int8, np.int16, np.int32, np.int64]

    # Test numerical preprocessing
    num_data = tabular_dataset._preprocess_numerical(data)
    assert num_data["feature_2"].mean() == pytest.approx(0, abs=1e-7)
    assert num_data["feature_2"].std() == pytest.approx(1, abs=1e-7)

    # Test preprocessing with zero variance
    bad_data = pd.DataFrame({"feature_2": [1.0, 1.0, 1.0, 1.0]})
    with pytest.raises(ValueError, match="has zero variance"):
        tabular_dataset._preprocess_numerical(bad_data)


def test_tabular_dataset_getitem(tabular_dataset):
    """Test TabularDataset __getitem__."""
    data, target, sensitive = tabular_dataset[0]
    assert data.shape == (10,)
    assert isinstance(target, torch.Tensor)
    assert sensitive.shape == (2,)


def test_dataset_transforms():
    """Test dataset transforms."""

    # Create simple transforms
    def feature_transform(x):
        return x * 2

    def target_transform(x):
        return x + 1

    def sensitive_transform(x):
        return x.float()

    # Create both datasets with same seed
    original_dataset = DummyFairDataset()
    transformed_dataset = DummyFairDataset()

    # Apply transforms to one dataset
    transformed_dataset.feature_transform = feature_transform
    transformed_dataset.target_transform = target_transform
    transformed_dataset.sensitive_transform = sensitive_transform

    # Get data from both datasets
    transformed_data, transformed_target, transformed_sensitive = transformed_dataset[0]
    original_data, original_target, original_sensitive = original_dataset[0]

    # Test transforms
    assert torch.allclose(transformed_data, original_data * 2)
    assert transformed_target == original_target + 1
    assert transformed_sensitive.dtype == torch.float


def test_attribute_selection():
    """Test target and sensitive attribute selection."""
    dataset = DummyFairDataset(
        target_attribute="employment", sensitive_attribute="race"
    )
    assert dataset.target_attribute == "employment"
    assert dataset.sensitive_attribute == "race"
    assert dataset.target_attribute in dataset.target_attribute_names
    assert dataset.sensitive_attribute in dataset.sensitive_attribute_names
