"""Implementation of Bank Marketing dataset for fairness research."""

import os
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.request
import zipfile
import gzip
import torch

from .base import TabularDataset


class BankMarketingDataset(TabularDataset):
    """Bank Marketing Dataset.

    This dataset contains direct marketing campaigns (phone calls) of a Portuguese banking institution.
    The task is to predict if the client will subscribe to a term deposit.
    Sensitive attributes include 'age', 'marital', and 'education'.

    Reference:
    [Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the
    Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014
    """

    _URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
    _FILES = {
        "bank-additional.csv": {
            "hash": "md5:f6cb2c1256ffe2836b36df321f46e92c",
            "note": "Dataset file",
        },
        "bank-additional-full.csv": {
            "hash": "md5:aec0451bc97d21b70e7acf88eb22448d",
            "note": "Dataset file",
        },
        "bank-additional-names.txt": {
            "hash": "md5:ee09b855dd692f74099ac5cba74f3e54",
            "note": "Dataset file",
        },
    }

    def __init__(
        self,
        root: str,
        split: str = "full",  # 'full' or 'reduced'
        download: bool = True,
        target_attribute: str = "y",
        sensitive_attribute: str = "age",
        feature_transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        sensitive_transform: Optional[Any] = None,
        categorical_encoding: str = "label",
        numerical_scaling: str = "standard",
    ):
        """Initialize the Bank Marketing dataset.

        Args:
            root: Root directory for the dataset
            split: Which version to use ('full' or 'reduced')
            download: Whether to download the dataset if not present
            target_attribute: Target attribute to use (defaults to 'y')
            sensitive_attribute: Sensitive attribute to use (defaults to 'age')
            feature_transform: Transform to apply to features
            target_transform: Transform to apply to targets
            sensitive_transform: Transform to apply to sensitive attributes
            categorical_encoding: Method for encoding categorical variables
            numerical_scaling: Method for scaling numerical variables
        """
        if split not in ["full", "reduced"]:
            raise ValueError(f"Invalid split: {split}. Must be 'full' or 'reduced'.")
        self.split = split
        self.base_folder = "bank-additional"

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
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "month",
            "day_of_week",
            "duration",
            "campaign",
            "pdays",
            "previous",
            "poutcome",
            "emp.var.rate",
            "cons.price.idx",
            "cons.conf.idx",
            "euribor3m",
            "nr.employed",
        ]

    @property
    def categorical_features(self) -> List[str]:
        """List of categorical feature names."""
        return [
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "month",
            "day_of_week",
            "poutcome",
        ]

    @property
    def numerical_features(self) -> List[str]:
        """List of numerical feature names."""
        return [
            "age",
            "duration",
            "campaign",
            "pdays",
            "previous",
            "emp.var.rate",
            "cons.price.idx",
            "cons.conf.idx",
            "euribor3m",
            "nr.employed",
        ]

    @property
    def sensitive_attribute_names(self) -> List[str]:
        """List of sensitive attribute names."""
        return ["age", "marital", "education"]

    @property
    def target_attribute_names(self) -> List[str]:
        """List of target attribute names."""
        return ["y"]

    @property
    def protected_groups(self) -> Dict[str, List[Any]]:
        """Get protected group values for each sensitive attribute."""
        return {
            "age": [">=60"],  # Age is often binarized at 60 for this dataset
            "marital": ["divorced", "single"],
            "education": ["basic.4y", "basic.6y", "basic.9y"],  # Lower education levels
        }

    @property
    def favourable_outcomes(self) -> Dict[str, List[Any]]:
        """Get favorable outcome values for each target attribute."""
        return {"y": ["yes"]}  # 'yes' means client subscribed to term deposit

    @property
    def recommended_metrics(self) -> List[str]:
        """Get recommended fairness metrics for this dataset."""
        return ["statistical_parity", "equal_opportunity", "disparate_impact"]

    @property
    def bias_notes(self) -> str:
        """Get notes about known biases in the dataset."""
        return """
        The Bank Marketing dataset may contain several types of biases:
        1. Age discrimination: Different treatment based on age groups
        2. Educational bias: Potential discrimination based on education level
        3. Marital status bias: Different treatment based on marital status
        4. Selection bias: Data collected only from one Portuguese bank
        5. Temporal bias: Campaign conducted during a specific time period
        6. Socioeconomic bias: May reflect economic inequalities
        7. Contact method bias: Phone-based marketing may not reach all demographics equally
        
        Note: This dataset represents real-world marketing practices which may
        reflect societal biases in financial services accessibility.
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

        # Download and extract zip file
        zip_path = os.path.join(self.root, self.base_folder, "bank-additional.zip")
        urllib.request.urlretrieve(self._URL, zip_path)

        # Extract the zip file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.root)

    def _load_data(self) -> None:
        """Load the dataset into memory."""
        filename = (
            "bank-additional-full.csv"
            if self.split == "full"
            else "bank-additional.csv"
        )
        file_path = os.path.join(self.root, self.base_folder, filename)

        # Load data
        self.data = pd.read_csv(file_path, delimiter=";")

        # Convert target to binary
        self.data["y"] = (self.data["y"] == "yes").astype(int)

        # Create age categories
        self.data["age_cat"] = pd.cut(
            self.data["age"],
            bins=[0, 25, 45, 60, float("inf")],
            labels=["<25", "25-45", "45-60", ">60"],
        )

        # Preprocess features
        features = self.data[self.feature_names].copy()
        features = self._preprocess_categorical(features)
        features = self._preprocess_numerical(features)

        # Convert numpy arrays to torch tensors
        self.features = torch.tensor(
            features.values.astype(np.float32), dtype=torch.float32
        )
        self.target = torch.tensor(
            self.data[self.target_attribute].values.astype(np.float32),
            dtype=torch.float32,
        )

        # Extract and encode sensitive attribute
        sensitive_data = self.data[self.sensitive_attribute].copy()
        if self.sensitive_attribute in self.categorical_features:
            # For categorical sensitive attributes, encode them first
            sensitive_data = pd.get_dummies(sensitive_data).iloc[:, 0]

        # Extract sensitive attribute and reshape to column vector
        self.sensitive = torch.tensor(
            sensitive_data.values.reshape(-1, 1).astype(np.float32),
            dtype=torch.float32,
        )
