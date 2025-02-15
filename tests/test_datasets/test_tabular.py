"""Tests for tabular datasets."""

import os
import pytest
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Any

from open_trustai.datasets.base import FairDataset, TabularDataset, VisionDataset
from open_trustai.datasets.adult import AdultDataset
from open_trustai.datasets.germancredit import GermanCreditDataset
from open_trustai.datasets.compas import CompasDataset
from open_trustai.datasets.bankmarketing import BankMarketingDataset


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


class TestGermanCreditDataset:
    def test_german_credit_dataset_init(self, tmp_path):
        """Test German Credit dataset initialization."""
        dataset = GermanCreditDataset(
            root=str(tmp_path), download=True, categorical_encoding="label"
        )
        assert len(dataset.feature_names) == 20
        assert "age" in dataset.sensitive_attribute_names
        assert "sex" in dataset.sensitive_attribute_names
        assert "foreign_worker" in dataset.sensitive_attribute_names

    def test_german_credit_dataset_loading(self, tmp_path):
        """Test German Credit dataset loading."""
        dataset = GermanCreditDataset(
            root=str(tmp_path), download=True, categorical_encoding="label"
        )
        assert len(dataset) == 1000  # German Credit dataset has 1000 samples

        # Test data loading
        data, target, sensitive = dataset[0]
        assert isinstance(data, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert isinstance(sensitive, torch.Tensor)

        # Check shapes
        assert len(data.shape) == 1
        assert target.shape == torch.Size([])
        assert len(sensitive.shape) == 1

    def test_german_credit_fairness_info(self, tmp_path):
        """Test German Credit dataset fairness information."""
        dataset = GermanCreditDataset(root=str(tmp_path), download=True)
        info = dataset.get_fairness_info()

        assert "age" in info["sensitive_attributes"]
        assert "sex" in info["sensitive_attributes"]
        assert "foreign_worker" in info["sensitive_attributes"]

        assert ">=25" in info["protected_groups"]["age"]
        assert "Female" in info["protected_groups"]["sex"]
        assert "Yes" in info["protected_groups"]["foreign_worker"]


class TestCompasDataset:
    def test_compas_dataset_init(self, tmp_path):
        """Test COMPAS dataset initialization."""
        dataset = CompasDataset(
            root=str(tmp_path), download=True, categorical_encoding="label"
        )
        assert len(dataset.feature_names) == 15
        assert "race" in dataset.sensitive_attribute_names
        assert "sex" in dataset.sensitive_attribute_names
        assert "age_cat" in dataset.sensitive_attribute_names

    def test_compas_dataset_loading(self, tmp_path):
        """Test COMPAS dataset loading."""
        dataset = CompasDataset(
            root=str(tmp_path), download=True, categorical_encoding="label"
        )
        assert (
            len(dataset) > 5000
        )  # COMPAS dataset has over 5000 samples after filtering

        # Test data loading
        data, target, sensitive = dataset[0]
        assert isinstance(data, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert isinstance(sensitive, torch.Tensor)

        # Check shapes
        assert len(data.shape) == 1
        assert target.shape == torch.Size([])
        assert len(sensitive.shape) == 1

    def test_compas_fairness_info(self, tmp_path):
        """Test COMPAS dataset fairness information."""
        dataset = CompasDataset(root=str(tmp_path), download=True)
        info = dataset.get_fairness_info()

        assert "race" in info["sensitive_attributes"]
        assert "sex" in info["sensitive_attributes"]
        assert "age_cat" in info["sensitive_attributes"]

        assert "African-American" in info["protected_groups"]["race"]
        assert "Female" in info["protected_groups"]["sex"]
        assert any("45" in age for age in info["protected_groups"]["age_cat"])


class TestBankMarketingDataset:
    def test_bank_marketing_dataset_init(self, tmp_path):
        """Test Bank Marketing dataset initialization."""
        dataset = BankMarketingDataset(
            root=str(tmp_path), download=True, categorical_encoding="label"
        )
        assert len(dataset.feature_names) == 20
        assert "age" in dataset.sensitive_attribute_names
        assert "marital" in dataset.sensitive_attribute_names
        assert "education" in dataset.sensitive_attribute_names

    def test_bank_marketing_dataset_loading(self, tmp_path):
        """Test Bank Marketing dataset loading."""
        dataset = BankMarketingDataset(
            root=str(tmp_path),
            download=True,
            split="full",
            categorical_encoding="label",
        )
        assert len(dataset) > 40000  # Full dataset has over 40k samples

        # Test data loading
        data, target, sensitive = dataset[0]
        assert isinstance(data, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert isinstance(sensitive, torch.Tensor)

        # Check shapes
        assert len(data.shape) == 1
        assert target.shape == torch.Size([])
        assert len(sensitive.shape) == 1

    def test_bank_marketing_splits(self, tmp_path):
        """Test Bank Marketing dataset splits."""
        full_dataset = BankMarketingDataset(
            root=str(tmp_path), split="full", download=True
        )
        assert len(full_dataset) > 40000

        reduced_dataset = BankMarketingDataset(
            root=str(tmp_path), split="reduced", download=True
        )
        assert len(reduced_dataset) > 4000  # Reduced dataset has over 4k samples

    def test_bank_marketing_fairness_info(self, tmp_path):
        """Test Bank Marketing dataset fairness information."""
        dataset = BankMarketingDataset(root=str(tmp_path), download=True)
        info = dataset.get_fairness_info()

        assert "age" in info["sensitive_attributes"]
        assert "marital" in info["sensitive_attributes"]
        assert "education" in info["sensitive_attributes"]

        assert ">=60" in info["protected_groups"]["age"]
        assert "divorced" in info["protected_groups"]["marital"]
        assert any("basic" in edu for edu in info["protected_groups"]["education"])
