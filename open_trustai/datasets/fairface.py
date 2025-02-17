"""Implementation of FairFace dataset for fairness research."""

import os
import zipfile
import pandas as pd
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple

# Add import error handling
try:
    import torch
    import gdown
    from torchvision import transforms
    from torchvision.datasets import CelebA as TorchCelebA

    _HAS_VISION_DEPS = True
except ImportError:
    _HAS_VISION_DEPS = False

from .base import VisionDataset


class FairFaceDataset(VisionDataset):
    """FairFace Dataset.

    FairFace is a face image dataset designed to fairly represent different races.
    It contains 108,501 images with race, gender, and age annotations, collected
    from the YFCC-100M Flickr dataset. Images were manually annotated with seven
    race groups, gender, and age groups.

    Sensitive attributes include 'race', 'gender', and 'age'.

    Reference:
    Kärkkäinen, K., & Joo, J. (2021). FairFace: Face Attribute Dataset for
    Balanced Race, Gender, and Age for Bias Measurement and Mitigation.
    In Proceedings of the IEEE/CVF Winter Conference on Applications of
    Computer Vision (pp. 1548-1558).
    """

    _FILES = {
        "fairface-img-margin025-trainval.zip": {
            "id": "1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86",
            "hash": "md5:16f1d00db9e17b11141047f0818c01c6",
        },
        "fairface_label_train.csv": {
            "id": "1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH",
            "hash": "md5:2ee4f612633640d7af3d9e51ba59b57e",
        },
        "fairface_label_val.csv": {
            "id": "1wOdja-ezstMEp81tX1a-EYkFebev4h7D",
            "hash": "md5:887a2051b2656427bdb21bbfdbb3e733",
        },
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        download: bool = True,
        target_attribute: str = "gender",
        sensitive_attribute: str = "race",
        feature_transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        sensitive_transform: Optional[Any] = None,
        image_size: Tuple[int, int] = (224, 224),
        cache_images: bool = True,
    ):
        """Initialize the FairFace dataset.

        Args:
            root: Root directory for the dataset
            split: Which split to use ('train' or 'valid')
            download: Whether to download the dataset if not present
            target_attribute: Target attribute to use (defaults to 'gender')
            sensitive_attribute: Sensitive attribute to use (defaults to 'race')
            feature_transform: Transform to apply to images
            target_transform: Transform to apply to target
            sensitive_transform: Transform to apply to sensitive attributes
            image_size: Size to resize images to (height, width)
            cache_images: Whether to cache images in memory
        """
        if not _HAS_VISION_DEPS:
            raise ImportError(
                "Vision dependencies not found. Please install them with:\n"
                "pip install torch torchvision Pillow gdown"
            )

        if split not in ["train", "valid"]:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'valid'.")
        self.base_folder = "fairface"
        self.split = "train" if split == "train" else "val"

        if feature_transform is None:
            feature_transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        super().__init__(
            root=root,
            download=download,
            target_attribute=target_attribute,
            sensitive_attribute=sensitive_attribute,
            feature_transform=feature_transform,
            target_transform=target_transform,
            sensitive_transform=sensitive_transform,
            image_size=image_size,
            cache_images=cache_images,
        )

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single example from the dataset.

        Returns:
            Tuple of (image, target, sensitive)
        """
        img_path = os.path.join(self.root, self.base_folder, self.df.iloc[idx]["file"])

        # Use image cache if enabled
        if self.cache_images and img_path in self._image_cache:
            image = self._image_cache[img_path]
        else:
            image = Image.open(img_path).convert("RGB")
            if self.cache_images:
                self._image_cache[img_path] = image

        if self.feature_transform:
            image = self.feature_transform(image)

        target = self.target[idx]
        sensitive = self.sensitive[idx]

        if self.target_transform:
            target = self.target_transform(target)
        if self.sensitive_transform:
            sensitive = self.sensitive_transform(sensitive)

        return image, target, sensitive

    @property
    def feature_names(self) -> List[str]:
        """List of feature names (not used for vision datasets)."""
        return []

    @property
    def sensitive_attribute_names(self) -> List[str]:
        """List of sensitive attribute names."""
        return ["race", "gender", "age"]

    @property
    def target_attribute_names(self) -> List[str]:
        """List of target attribute names."""
        return ["gender", "race", "age"]

    @property
    def protected_groups(self) -> Dict[str, List[Any]]:
        """Get protected group values for each sensitive attribute."""
        return {
            "race": [1, 2, 3],  # Non-white races
            "gender": [1],  # Female
            "age": [0, 1, 2, 6, 7, 8],  # Very young and older age groups
        }

    @property
    def favourable_outcomes(self) -> Dict[str, List[Any]]:
        """Get favorable outcome values for each target attribute."""
        return {
            "race": [0],  # White
            "gender": [0],  # Male
            "age": [3, 4, 5],  # Middle age groups
        }

    @property
    def recommended_metrics(self) -> List[str]:
        """Get recommended fairness metrics for this dataset."""
        return ["statistical_parity", "equal_opportunity", "disparate_impact"]

    @property
    def bias_notes(self) -> str:
        """Get notes about known biases in the dataset."""
        return """
        The FairFace dataset, while designed for fairness, may still contain biases:
        1. Racial categorization bias: Simplified race categories may not capture full diversity
        2. Binary gender bias: Gender labels are binary (male/female)
        3. Age estimation bias: Age groups may have estimation errors
        4. Image quality bias: Variations in image quality across demographics
        5. Cultural bias: Western-centric categorizations of race and gender
        6. Annotation bias: Human annotators may have inherent biases
        7. Collection bias: Despite efforts for balance, some groups may be underrepresented
        
        Note: While FairFace improves upon previous datasets in terms of racial
        diversity, it still uses simplified categories that may not capture the
        full spectrum of human diversity.
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

        for filename, value in self._FILES.items():
            gdown.cached_download(
                url=f"https://drive.google.com/uc?id={value['id']}",
                path=os.path.join(self.root, self.base_folder, filename),
                hash=value["hash"],
                quiet=False,
            )

        # Extract image zip file
        with zipfile.ZipFile(
            os.path.join(
                self.root, self.base_folder, "fairface-img-margin025-trainval.zip"
            ),
            "r",
        ) as zf:
            zf.extractall(os.path.join(self.root, self.base_folder))

    def _load_data(self) -> None:
        """Load the dataset into memory."""
        csv_file = os.path.join(
            self.root, self.base_folder, f"fairface_label_{self.split}.csv"
        )

        if not os.path.exists(csv_file):
            raise RuntimeError(
                f"Dataset not found at {csv_file}. Please use download=True to download it."
            )

        self.df = pd.read_csv(csv_file)

        # Map string labels to integers
        label_maps = {
            "race": {"White": 0, "Black": 1, "Asian": 2, "Indian": 3},
            "gender": {"Male": 0, "Female": 1},
            "age": {
                "0-2": 0,
                "3-9": 1,
                "10-19": 2,
                "20-29": 3,
                "30-39": 4,
                "40-49": 5,
                "50-59": 6,
                "60-69": 7,
                "more than 70": 8,
            },
        }

        for attr, mapping in label_maps.items():
            self.df[attr] = self.df[attr].map(mapping)

        # Extract target and sensitive attributes
        # Binarize sensitive attributes based on selection
        sensitive_data = self.df[self.sensitive_attribute].copy()
        if self.sensitive_attribute == "gender":
            # Binarize gender: 0 if female, 1 if male
            sensitive_data = (sensitive_data != 1).astype(int)
        elif self.sensitive_attribute == "race":
            # Binarize race: 1 if white, 0 otherwise
            sensitive_data = (sensitive_data == 0).astype(int)
        elif self.sensitive_attribute == "age":
            # Binarize age: 0 if age < 30 (groups 0-29), 1 otherwise
            sensitive_data = (sensitive_data < 3).astype(int)

        self.df[self.sensitive_attribute] = sensitive_data
        self.target = torch.tensor(self.df[self.target_attribute].values)
        self.sensitive = torch.tensor(self.df[self.sensitive_attribute].values)
