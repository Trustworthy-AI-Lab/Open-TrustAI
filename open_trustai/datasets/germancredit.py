"""Implementation of German Credit dataset for fairness research."""

import os
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.request
import zipfile
import gzip
import torch

from .base import TabularDataset


class GermanCreditDataset(TabularDataset):
    """German Credit Dataset.

    The German Credit dataset contains credit data from a German bank.
    The task is to predict whether a customer has good or bad credit risk.
    Sensitive attributes include 'age', 'personal_status_sex', and 'foreign_worker'.

    Reference:
    Dua, D. and Graff, C. (2019). UCI Machine Learning Repository.
    Irvine, CA: University of California, School of Information and Computer Science.
    """

    _URL = (
        "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
    )
    _FILES = {
        "german.data": {
            "hash": "md5:6b94c2e35480e671545e52a808a8a549",
            "note": "Dataset file",
        },
        "german.doc": {
            "hash": "md5:9e5a5db935ce1ee6720d95ad614d873c",
            "note": "Documentation file",
        },
    }

    def __init__(
        self,
        root: str,
        download: bool = True,
        target_attribute: str = "credit_risk",
        sensitive_attribute: str = "personal_status_sex",
        feature_transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        sensitive_transform: Optional[Any] = None,
        categorical_encoding: str = "label",
        numerical_scaling: str = "standard",
    ):
        """Initialize the German Credit dataset.

        Args:
            root: Root directory for the dataset
            download: Whether to download the dataset if not present
            target_attribute: Target attribute to use (defaults to 'credit_risk')
            sensitive_attribute: Sensitive attribute to use (defaults to 'personal_status_sex')
            feature_transform: Transform to apply to features
            target_transform: Transform to apply to targets
            sensitive_transform: Transform to apply to sensitive attributes
            categorical_encoding: Method for encoding categorical variables
            numerical_scaling: Method for scaling numerical variables
        """

        self.base_folder = "german"

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
            "status",
            "duration",
            "credit_history",
            "purpose",
            "credit_amount",
            "savings",
            "employment_duration",
            "installment_rate",
            "personal_status_sex",
            "other_debtors",
            "present_residence",
            "property",
            "age",
            "other_installments",
            "housing",
            "existing_credits",
            "job",
            "num_dependents",
            "phone",
            "foreign_worker",
        ]

    @property
    def categorical_features(self) -> List[str]:
        """List of categorical feature names."""
        return [
            "status",
            "credit_history",
            "purpose",
            "savings",
            "employment_duration",
            "personal_status_sex",
            "other_debtors",
            "property",
            "other_installments",
            "housing",
            "job",
            "phone",
            "foreign_worker",
        ]

    @property
    def numerical_features(self) -> List[str]:
        """List of numerical feature names."""
        return [
            "duration",
            "credit_amount",
            "installment_rate",
            "present_residence",
            "age",
            "existing_credits",
            "num_dependents",
        ]

    @property
    def sensitive_attribute_names(self) -> List[str]:
        """List of sensitive attribute names."""
        return ["age", "personal_status_sex", "foreign_worker"]

    @property
    def target_attribute_names(self) -> List[str]:
        """List of target attribute names."""
        return ["credit_risk"]

    @property
    def protected_groups(self) -> Dict[str, List[Any]]:
        """Get protected group values for each sensitive attribute."""
        return {
            "age": [">=25"],  # Age is often binarized at 25
            "personal_status_sex": ["Female"],
            "foreign_worker": ["Yes"],
        }

    @property
    def favourable_outcomes(self) -> Dict[str, List[Any]]:
        """Get favorable outcome values for each target attribute."""
        return {"credit_risk": ["Good"]}

    @property
    def recommended_metrics(self) -> List[str]:
        """Get recommended fairness metrics for this dataset."""
        return ["statistical_parity", "equal_opportunity", "disparate_impact"]

    @property
    def bias_notes(self) -> str:
        """Get notes about known biases in the dataset."""
        return """
        The German Credit dataset contains several known biases:
        1. Gender bias: Different treatment based on gender and marital status
        2. Age bias: Age may unfairly influence credit decisions
        3. Nationality bias: Foreign workers may face discrimination
        4. Historical bias: The data reflects historical societal biases in lending
        5. Sampling bias: The dataset is from a specific time and region
        6. Institutional bias: Banking practices of the time may have been discriminatory
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
        zip_path = os.path.join(self.root, self.base_folder, "german.zip")
        urllib.request.urlretrieve(self._URL, zip_path)

        # Extract the zip file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(os.path.join(self.root, self.base_folder))

    def _load_data(self) -> None:
        """Load the dataset into memory."""
        file_path = os.path.join(self.root, self.base_folder, "german.data")

        # Load data
        self.df = pd.read_csv(file_path, sep=" ", header=None)

        # Assign column names
        self.df.columns = self.feature_names + ["credit_risk"]

        # Preprocess features
        features = self.df[self.feature_names].copy()
        features = self._preprocess_categorical(features)
        features = self._preprocess_numerical(features)

        # Convert numpy arrays to torch tensors
        self.features = torch.tensor(
            features.values.astype(np.float32), dtype=torch.float32
        )
        # normalize target from 1, 2 to 0, 1
        self.target = torch.tensor(
            (self.df[self.target_attribute].values - 1).astype(np.float32),
            dtype=torch.float32,
        )

        # Extract and encode sensitive attribute
        sensitive_data = self.df[self.sensitive_attribute].copy()
        if self.sensitive_attribute == "age":
            # Binarize age: 0 if age < 25, 1 otherwise
            sensitive_data = (sensitive_data < 25).astype(int)
        elif self.sensitive_attribute == "personal_status_sex":
            # Binarize personal_status_sex: 0 if Female, 1 if Male
            sensitive_data = (sensitive_data != "Female").astype(int)
        elif self.sensitive_attribute == "foreign_worker":
            # Binarize foreign_worker: 0 if Yes, 1 if No
            sensitive_data = (sensitive_data != "Yes").astype(int)
        elif self.sensitive_attribute == "education":
            # Binarize education: 1 if has higher education, 0 otherwise
            sensitive_data = (sensitive_data.isin(["high", "university"])).astype(int)

        # Extract sensitive attribute and reshape to column vector
        self.sensitive = torch.tensor(
            sensitive_data.values.astype(np.float32),
            dtype=torch.float32,
        ).reshape(-1, 1)
