"""Implementation of COMPAS dataset for fairness research."""

import os
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.request
import zipfile
import gzip
import torch

from .base import TabularDataset


class CompasDataset(TabularDataset):
    """COMPAS Recidivism Dataset.

    The COMPAS (Correctional Offender Management Profiling for Alternative Sanctions)
    dataset contains criminal history, jail and prison time, demographics, and COMPAS
    risk scores for defendants from Broward County, Florida. The task is to predict
    recidivism within two years.

    Sensitive attributes include 'race', 'sex', and 'age_cat'.

    Reference:
    ProPublica's COMPAS Analysis:
    https://github.com/propublica/compas-analysis
    """

    _URL = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    _FILES = {
        "compas-scores-two-years.csv": {
            "hash": "md5:9165d40c400bba93a8cffece2b74622b",
            "note": "Dataset file",
        },
    }

    def __init__(
        self,
        root: str,
        download: bool = True,
        target_attribute: str = "is_recid",
        sensitive_attribute: str = "race",
        feature_transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        sensitive_transform: Optional[Any] = None,
        categorical_encoding: str = "label",
        numerical_scaling: str = "standard",
    ):
        """Initialize the COMPAS dataset.

        Args:
            root: Root directory for the dataset
            download: Whether to download the dataset if not present
            target_attribute: Target attribute to use (defaults to 'is_recid')
            sensitive_attribute: Sensitive attribute to use (defaults to 'race')
            feature_transform: Transform to apply to features
            target_transform: Transform to apply to targets
            sensitive_transform: Transform to apply to sensitive attributes
            categorical_encoding: Method for encoding categorical variables
            numerical_scaling: Method for scaling numerical variables
        """
        self.base_folder = "compas"

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
            "sex",
            "race",
            "juv_fel_count",
            "juv_misd_count",
            "juv_other_count",
            "priors_count",
            "c_charge_degree",
            "days_b_screening_arrest",
            "c_days_from_compas",
            "decile_score",
            "age_cat",
            "score_text",
            "v_decile_score",
            "v_score_text",
        ]

    @property
    def categorical_features(self) -> List[str]:
        """List of categorical feature names."""
        return [
            "sex",
            "race",
            "c_charge_degree",
            "age_cat",
            "score_text",
            "v_score_text",
        ]

    @property
    def numerical_features(self) -> List[str]:
        """List of numerical feature names."""
        return [
            "age",
            "juv_fel_count",
            "juv_misd_count",
            "juv_other_count",
            "priors_count",
            "days_b_screening_arrest",
            "c_days_from_compas",
            "decile_score",
            "v_decile_score",
        ]

    @property
    def sensitive_attribute_names(self) -> List[str]:
        """List of sensitive attribute names."""
        return ["race", "sex", "age_cat"]

    @property
    def target_attribute_names(self) -> List[str]:
        """List of target attribute names."""
        return ["is_recid"]

    @property
    def protected_groups(self) -> Dict[str, List[Any]]:
        """Get protected group values for each sensitive attribute."""
        return {
            "race": ["African-American", "Hispanic"],
            "sex": ["Female"],
            "age_cat": ["Greater than 45", "25 - 45"],
        }

    @property
    def favourable_outcomes(self) -> Dict[str, List[Any]]:
        """Get favorable outcome values for each target attribute."""
        return {"is_recid": ["no_recid"]}

    @property
    def recommended_metrics(self) -> List[str]:
        """Get recommended fairness metrics for this dataset."""
        return [
            "statistical_parity",
            "equal_opportunity",
            "disparate_impact",
            "equalized_odds",
        ]

    @property
    def bias_notes(self) -> str:
        """Get notes about known biases in the dataset."""
        return """
        The COMPAS dataset contains several significant biases:
        1. Racial bias: ProPublica's analysis found that the algorithm was biased against African-American defendants
        2. Gender bias: Different false positive and false negative rates across genders
        3. Age bias: Age category influences risk scores
        4. Historical bias: Reflects historical discrimination in the criminal justice system
        5. Sampling bias: Data is from a specific county and time period
        6. Label bias: Recidivism prediction relies on arrest data, which may be biased
        7. Institutional bias: Reflects systemic biases in law enforcement and judicial practices
        
        Note: This dataset has been the subject of significant controversy and debate
        regarding algorithmic fairness in criminal justice applications.
        """

    def _check_exists(self) -> bool:
        """Check if the dataset exists in the root directory."""
        return os.path.exists(
            os.path.join(self.root, self.base_folder, "compas-scores-two-years.csv")
        )

    def _download(self) -> None:
        """Download the dataset if it doesn't exist."""
        if self._check_exists():
            return

        os.makedirs(self.root, exist_ok=True)
        os.makedirs(os.path.join(self.root, self.base_folder), exist_ok=True)

        # Download file
        target_path = os.path.join(
            self.root, self.base_folder, "compas-scores-two-years.csv"
        )
        urllib.request.urlretrieve(self._URL, target_path)

    def _load_data(self) -> None:
        """Load the dataset into memory."""
        file_path = os.path.join(
            self.root, self.base_folder, "compas-scores-two-years.csv"
        )

        # Load data
        self.data = pd.read_csv(file_path)

        # Filter the data according to ProPublica's analysis
        self.data = self.data[
            (self.data.days_b_screening_arrest <= 30)
            & (self.data.days_b_screening_arrest >= -30)
            & (self.data.is_recid != -1)
            & (self.data.c_charge_degree != "O")
            & (self.data.score_text != "N/A")
        ]

        # Ensure all categorical features are strings
        for cat_feature in self.categorical_features:
            self.data[cat_feature] = self.data[cat_feature].astype(str)

        # Handle missing values
        for num_feature in self.numerical_features:
            self.data[num_feature] = self.data[num_feature].fillna(
                self.data[num_feature].mean()
            )
        for cat_feature in self.categorical_features:
            self.data[cat_feature] = self.data[cat_feature].fillna(
                self.data[cat_feature].mode()[0]
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
            self.data["is_recid"].values.astype(np.float32),
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
