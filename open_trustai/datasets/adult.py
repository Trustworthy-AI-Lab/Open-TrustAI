"""Implementation of Adult Income dataset for fairness research."""

import os
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.request
import zipfile
import gzip
import torch

from .base import TabularDataset


class AdultDataset(TabularDataset):
    """Adult Income Dataset.

    The Adult Income dataset contains census data and the task is to predict
    whether a person's income is above $50K/year based on census data.
    Sensitive attributes include 'sex', 'race', and 'age'.

    Reference:
    Dua, D. and Graff, C. (2019). UCI Machine Learning Repository.
    Irvine, CA: University of California, School of Information and Computer Science.
    """

    _URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
    _FILES = {
        "adult.data": {
            "hash": "md5:5d7c39d7b8804f071cdd1f2a7c460872",
            "note": "Training data",
        },
        "adult.test": {
            "hash": "md5:1a7cdb3ff7a1b709968b1c7a11def63e",
            "note": "Test data",
        },
        "adult.names": {
            "hash": "md5:35238206dfdf7f1fe215bbb874adecdc",
            "note": "Feature names",
        },
    }

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        download: bool = False,
        target_attribute: str = "income",
        sensitive_attribute: str = "sex",
        feature_transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        sensitive_transform: Optional[Any] = None,
        categorical_encoding: str = "label",
        numerical_scaling: str = "standard",
    ):
        """Initialize the Adult dataset.

        Args:
            root: Root directory for the dataset
            split: Which split to use ('train' or 'test')
            download: Whether to download the dataset if not present
            target_attribute: Target attribute to use (defaults to 'income')
            sensitive_attribute: Sensitive attribute to use (if None, uses 'sex')
            feature_transform: Transform to apply to features
            target_transform: Transform to apply to targets
            sensitive_transform: Transform to apply to sensitive attributes
            categorical_encoding: Method for encoding categorical variables
            numerical_scaling: Method for scaling numerical variables
        """
        if split not in ["train", "test"]:
            raise ValueError(f"Invalid split. Must be one of 'train' or 'test'")
        self.split = split
        self.base_folder = "adult"

        super().__init__(
            root=root,
            download=download,
            target_attribute=target_attribute,
            sensitive_attribute=sensitive_attribute,
            feature_transform=feature_transform,
            target_transform=target_transform,
            sensitive_transform=sensitive_transform,
            categorical_encoding=categorical_encoding,
            numerical_scaling=numerical_scaling,
        )

    @property
    def feature_names(self) -> List[str]:
        """List of feature names."""
        return [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
        ]

    @property
    def categorical_features(self) -> List[str]:
        """List of categorical feature names."""
        return [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]

    @property
    def numerical_features(self) -> List[str]:
        """List of numerical feature names."""
        return [
            "age",
            "fnlwgt",
            "education-num",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
        ]

    @property
    def sensitive_attribute_names(self) -> List[str]:
        """List of sensitive attribute names."""
        return ["sex", "race", "age"]

    @property
    def target_attribute_names(self) -> List[str]:
        """List of target attribute names."""
        return ["income"]

    @property
    def protected_groups(self) -> Dict[str, List[Any]]:
        """Get protected group values for each sensitive attribute."""
        return {
            "sex": ["Female"],
            "race": ["Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
            "age": [">=40"],  # Age is often binarized at 40
        }

    @property
    def favourable_outcomes(self) -> Dict[str, List[Any]]:
        """Get favorable outcome values for each target attribute."""
        return {"income": [">50K"]}

    @property
    def recommended_metrics(self) -> List[str]:
        """Get recommended fairness metrics for this dataset."""
        return ["statistical_parity", "equal_opportunity", "disparate_impact"]

    @property
    def bias_notes(self) -> str:
        """Get notes about known biases in the dataset."""
        return """
        The Adult Income dataset contains several known biases:
        1. Gender bias: Women are underrepresented in high-income groups
        2. Racial bias: Minorities are underrepresented in high-income groups
        3. Age bias: Younger individuals are underrepresented in high-income groups
        4. Historical bias: The data reflects historical societal biases in income distribution
        5. Sampling bias: The dataset is from the 1994 US Census and may not reflect current demographics
        """

    def _check_exists(self) -> bool:
        """Check if the dataset exists in the root directory."""
        return all(
            os.path.exists(os.path.join(self.root, self.base_folder, f))
            for f in self._FILES.keys()
        )

    def _download(self) -> None:
        """Download the dataset if it doesn't exist."""
        if self._check_exists():
            return

        os.makedirs(self.root, exist_ok=True)
        os.makedirs(os.path.join(self.root, self.base_folder), exist_ok=True)

        # Download zip file
        zip_path = os.path.join(self.root, self.base_folder, "adult.zip")
        urllib.request.urlretrieve(self._URL, zip_path)

        # Extract the zip file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(os.path.join(self.root, self.base_folder))

    def _load_data(self) -> None:
        """Load the dataset into memory."""
        file_path = os.path.join(
            self.root,
            self.base_folder,
            "adult.data" if self.split == "train" else "adult.test",
        )

        # Define column names
        columns = self.feature_names + ["income"]

        # Load data
        # The test set has a different format with dots at the end of each line
        if self.split == "test":
            # Load test data and remove trailing periods
            self.df = pd.read_csv(
                file_path,
                names=columns,
                skipinitialspace=True,
                na_values=["?"],
                skiprows=1,
            )
            self.df = self.df.apply(
                lambda x: x.str.rstrip(".") if x.dtype == "object" else x
            )
        else:
            self.df = pd.read_csv(
                file_path, names=columns, skipinitialspace=True, na_values=["?"]
            )

        # Clean income labels
        self.df["income"] = self.df["income"].map(
            lambda x: 1 if x.strip().startswith(">50K") else 0
        )

        # Handle missing values
        for col in self.categorical_features:
            self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        for col in self.numerical_features:
            self.df[col].fillna(self.df[col].mean(), inplace=True)

        # Preprocess features
        features = self.df[self.feature_names].copy()
        features = self._preprocess_categorical(features)
        features = self._preprocess_numerical(features)

        # Convert numpy arrays to torch tensors
        self.features = torch.tensor(
            features.values.astype(np.float32), dtype=torch.float32
        )
        self.target = torch.tensor(
            self.df[self.target_attribute].values.astype(np.float32),
            dtype=torch.float32,
        )

        # Extract and encode sensitive attribute
        sensitive_data = self.df[self.sensitive_attribute].copy()
        if self.sensitive_attribute == "sex":
            # Binarize sex: 0 if Female, 1 if Male
            sensitive_data = (sensitive_data == "Male").astype(int)
        elif self.sensitive_attribute == "race":
            # Binarize race: 1 if White, 0 otherwise
            sensitive_data = (sensitive_data == "White").astype(int)
        elif self.sensitive_attribute == "age":
            # Binarize age: 0 if age >= 40, 1 otherwise
            sensitive_data = (sensitive_data < 40).astype(int)
        elif self.sensitive_attribute == "marital-status":
            # Binarize marital status: 0 if married, 1 otherwise
            sensitive_data = (sensitive_data != "Married-civ-spouse").astype(int)
        elif self.sensitive_attribute == "education":
            # Binarize education: 1 if has college degree, 0 otherwise
            sensitive_data = (
                sensitive_data.isin(["Bachelors", "Masters", "Doctorate"])
            ).astype(int)

        # Extract sensitive attribute and reshape to column vector
        self.sensitive = torch.tensor(
            sensitive_data.values.astype(np.float32),
            dtype=torch.float32,
        ).reshape(-1, 1)
