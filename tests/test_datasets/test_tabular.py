"""Tests for tabular datasets."""

import os
import pytest
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Any

from open_trustai.datasets.tabular import (
    AdultDataset,
    GermanCreditDataset,
    CompasDataset,
    BankMarketingDataset,
)


class TestAdultDataset:
    def test_adult_dataset_init(self, tmp_path):
        """Test Adult dataset initialization."""
        adult_dataset = AdultDataset(
            root=str(tmp_path),
            download=True,
            categorical_encoding="label",
            numerical_scaling="standard",
        )
        assert len(adult_dataset.feature_names) == 14
        assert "sex" in adult_dataset.sensitive_attribute_names
        assert "race" in adult_dataset.sensitive_attribute_names
        assert "age" in adult_dataset.sensitive_attribute_names

    def test_adult_dataset_loading(self, tmp_path):
        """Test Adult dataset loading."""
        adult_dataset = AdultDataset(
            root=str(tmp_path),
            split="train",
            download=True,
            categorical_encoding="label",
            numerical_scaling="standard",
        )
        assert len(adult_dataset) == 32561

        # Test data loading
        data, target, sensitive = adult_dataset[0]
        assert isinstance(data, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert isinstance(sensitive, torch.Tensor)

        # Check shapes
        assert len(data.shape) == 1  # 1D feature vector
        assert target.shape == torch.Size([])  # Single label
        assert len(sensitive.shape) == 1  # 1D sensitive attributes vector

    def test_adult_dataset_fairness_info(self, tmp_path):
        """Test Adult dataset fairness information."""
        adult_dataset = AdultDataset(
            root=str(tmp_path),
            download=True,
            categorical_encoding="label",
            numerical_scaling="standard",
        )
        info = adult_dataset.get_fairness_info()

        # Check structure and content
        assert "sensitive_attributes" in info
        assert "protected_groups" in info
        assert "fairness_metrics" in info
        assert "bias_notes" in info

        assert "sex" in info["sensitive_attributes"]
        assert "race" in info["sensitive_attributes"]
        assert "age" in info["sensitive_attributes"]

        protected_groups = info["protected_groups"]
        assert "Female" in protected_groups["sex"]
        assert any("Black" in race for race in protected_groups["race"])
        assert ">=40" in protected_groups["age"]

    def test_adult_dataset_splits(self, tmp_path):
        """Test Adult dataset train/test splits."""
        train_dataset = AdultDataset(root=str(tmp_path), split="train", download=True)
        assert len(train_dataset) == 32561

        test_dataset = AdultDataset(root=str(tmp_path), split="test", download=True)
        assert len(test_dataset) == 16281

    def test_adult_dataset_invalid_params(self, tmp_path):
        """Test Adult dataset with invalid parameters."""
        with pytest.raises(ValueError):
            AdultDataset(
                root=str(tmp_path),
                download=True,
                categorical_encoding="invalid_encoding",
            )

        with pytest.raises(ValueError):
            AdultDataset(
                root=str(tmp_path), download=True, numerical_scaling="invalid_scaling"
            )

        with pytest.raises(ValueError):
            AdultDataset(root=str(tmp_path), download=True, split="invalid_split")

    def test_adult_dataset_target_space(self, tmp_path):
        """Test that train and test splits have the same target space."""
        train_dataset = AdultDataset(root=str(tmp_path), split="train", download=True)
        test_dataset = AdultDataset(root=str(tmp_path), split="test", download=True)

        # Get unique target values from both splits
        train_targets = torch.unique(train_dataset.target).tolist()
        test_targets = torch.unique(test_dataset.target).tolist()

        # Sort both lists to ensure consistent comparison
        train_targets.sort()
        test_targets.sort()

        # Check that target spaces are identical
        assert (
            train_targets == test_targets
        ), "Train and test splits have different target spaces"
