"""Datasets module for Open-TrustAI."""

from open_trustai.datasets.base import FairDataset, TabularDataset, VisionDataset
from open_trustai.datasets.adult import AdultDataset
from open_trustai.datasets.germancredit import GermanCreditDataset
from open_trustai.datasets.compas import CompasDataset
from open_trustai.datasets.bankmarketing import BankMarketingDataset
from open_trustai.datasets.celeba import CelebADataset
from open_trustai.datasets.utkface import UTKFaceDataset
from open_trustai.datasets.fairface import FairFaceDataset

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
