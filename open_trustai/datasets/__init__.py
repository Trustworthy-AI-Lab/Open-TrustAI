"""Datasets module for Open-TrustAI."""

from open_trustai.datasets.base import FairDataset, TabularDataset, VisionDataset
from open_trustai.datasets.tabular import (
    AdultDataset,
    GermanCreditDataset,
    CompasDataset,
    BankMarketingDataset,
)
from open_trustai.datasets.vision import CelebADataset, UTKFaceDataset, FairFaceDataset

__all__ = [
    # Base classes
    "FairDataset",
    "TabularDataset",
    "VisionDataset",
    # Tabular datasets
    "AdultDataset",
    "GermanCreditDataset",
    "CompasDataset",
    "BankMarketingDataset",
    # Vision datasets
    "CelebADataset",
    "UTKFaceDataset",
    "FairFaceDataset",
]
