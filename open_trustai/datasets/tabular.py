"""Implementation of tabular datasets for fairness research."""

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
            self.data = pd.read_csv(
                file_path,
                names=columns,
                skipinitialspace=True,
                na_values=["?"],
                skiprows=1,
            )
            self.data = self.data.apply(
                lambda x: x.str.rstrip(".") if x.dtype == "object" else x
            )
        else:
            self.data = pd.read_csv(
                file_path, names=columns, skipinitialspace=True, na_values=["?"]
            )

        # Clean income labels
        self.data["income"] = self.data["income"].map(
            lambda x: 1 if x.strip().startswith(">50K") else 0
        )

        # Handle missing values
        for col in self.categorical_features:
            self.data[col].fillna(self.data[col].mode()[0], inplace=True)
        for col in self.numerical_features:
            self.data[col].fillna(self.data[col].mean(), inplace=True)

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


class GermanCreditDataset(TabularDataset):
    """German Credit Dataset.

    The German Credit dataset contains credit data from a German bank.
    The task is to predict whether a customer has good or bad credit risk.
    Sensitive attributes include 'age', 'sex', and 'foreign_worker'.

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
        sensitive_attribute: str = "sex",
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
            sensitive_attribute: Sensitive attribute to use (defaults to 'sex')
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
        return ["age", "sex", "foreign_worker"]

    @property
    def target_attribute_names(self) -> List[str]:
        """List of target attribute names."""
        return ["credit_risk"]

    @property
    def protected_groups(self) -> Dict[str, List[Any]]:
        """Get protected group values for each sensitive attribute."""
        return {
            "age": [">=25"],  # Age is often binarized at 25
            "sex": ["Female"],
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
        self.data = pd.read_csv(file_path, sep=" ", header=None)

        # Assign column names
        self.data.columns = self.feature_names + ["credit_risk"]

        # Extract sensitive attributes
        self.data["sex"] = self.data["personal_status_sex"].map(
            lambda x: 1 if x in [2, 3, 4] else 0  # Female = 1, Male = 0
        )

        # Convert target to binary (1 = good, 0 = bad)
        self.data["credit_risk"] = self.data["credit_risk"].map(
            lambda x: 1 if x == 1 else 0
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
