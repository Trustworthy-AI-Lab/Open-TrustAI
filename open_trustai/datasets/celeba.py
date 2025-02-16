"""Implementation of CelebA dataset for fairness research."""

import os
import pandas as pd
from PIL import Image
import zipfile
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


class CelebADataset(VisionDataset):
    """CelebA Dataset.

    CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset
    with more than 200K celebrity images, each with 40 attribute annotations.
    The images cover large pose variations and background clutter.

    Sensitive attributes include 'Male', 'Young', 'Attractive', etc.

    Reference:
    Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep Learning Face Attributes
    in the Wild. In Proceedings of International Conference on Computer Vision (ICCV).
    """

    _FILES = {
        "img_align_celeba.zip": {
            "id": "0B7EVK8r0v71pZjFTYXZWM3FlRnM",
            "hash": "md5:00d2c5bc6d35e252742224ab0c1e8fcb",
        },
        "list_attr_celeba.txt": {
            "id": "0B7EVK8r0v71pblRyaVFSWGxPY0U",
            "hash": "md5:75e246fa4810816ffd6ee81facbd244c",
        },
        "identity_CelebA.txt": {
            "id": "1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS",
            "hash": "md5:32bd1bd63d3c78cd57e08160ec5ed1e2",
        },
        "list_bbox_celeba.txt": {
            "id": "0B7EVK8r0v71pbThiMVRxWXZ4dU0",
            "hash": "md5:00566efa6fedff7a56946cd1c10f1c16",
        },
        "list_landmarks_align_celeba.txt": {
            "id": "0B7EVK8r0v71pd0FJY3Blby1HUTQ",
            "hash": "md5:cc24ecafdb5b50baae59b03474781f8c",
        },
        "list_eval_partition.txt": {
            "id": "0B7EVK8r0v71pY0NSMzRuSXJEVkk",
            "hash": "md5:d32c9cbf5e040fd4025c592c306e6668",
        },
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        download: bool = True,
        target_attribute: str = "Attractive",
        sensitive_attribute: str = "Male",
        feature_transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        sensitive_transform: Optional[Any] = None,
        image_size: Tuple[int, int] = (224, 224),
        cache_images: bool = True,
    ):
        """Initialize the CelebA dataset.

        Args:
            root: Root directory for the dataset
            split: Which split to use ('train', 'valid', or 'test')
            download: Whether to download the dataset if not present
            target_attribute: Target attribute to use (defaults to 'Attractive')
            sensitive_attribute: Sensitive attribute to use (defaults to 'Male')
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

        if split not in ["train", "valid", "test"]:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train', 'valid', or 'test'."
            )
        self.base_folder = "celeba"
        self.split = split

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
        img_path = os.path.join(
            self.root,
            self.base_folder,
            "img_align_celeba",
            self.df.iloc[idx]["file"],
        )

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
        return ["Male", "Young", "Attractive"]

    @property
    def target_attribute_names(self) -> List[str]:
        """List of target attribute names."""
        return ["Attractive", "Male", "Young"]

    @property
    def protected_groups(self) -> Dict[str, List[Any]]:
        """Get protected group values for each sensitive attribute."""
        return {
            "Male": [0],  # Female
            "Young": [0],  # Not young
            "Attractive": [0],  # Not attractive
        }

    @property
    def favourable_outcomes(self) -> Dict[str, List[Any]]:
        """Get favorable outcome values for each target attribute."""
        return {
            "Attractive": [1],  # Attractive
            "Male": [1],  # Male
            "Young": [1],  # Young
        }

    @property
    def recommended_metrics(self) -> List[str]:
        """Get recommended fairness metrics for this dataset."""
        return ["statistical_parity", "equal_opportunity", "disparate_impact"]

    @property
    def bias_notes(self) -> str:
        """Get notes about known biases in the dataset."""
        return """
        The CelebA dataset contains several known biases:
        1. Gender bias: Imbalanced representation and stereotypical attributes
        2. Age bias: Underrepresentation of certain age groups
        3. Racial bias: Limited diversity in racial representation
        4. Beauty standards bias: Western-centric beauty standards
        5. Celebrity bias: Not representative of general population
        6. Annotation bias: Subjective attribute labels
        7. Quality bias: Professional photos in controlled settings
        
        Note: This dataset primarily contains celebrity images and may not
        reflect the diversity of the general population.
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
            os.path.join(self.root, self.base_folder, "img_align_celeba.zip"), "r"
        ) as zf:
            zf.extractall(os.path.join(self.root, self.base_folder))

    def _load_data(self) -> None:
        """Load the dataset into memory."""
        attr_file = os.path.join(self.root, self.base_folder, "list_attr_celeba.txt")
        partition_file = os.path.join(
            self.root, self.base_folder, "list_eval_partition.txt"
        )

        if not os.path.exists(attr_file):
            raise RuntimeError(
                f"Dataset not found at {attr_file}. Please use download=True to download it."
            )

        # Read attribute file
        self.df = pd.read_csv(attr_file, sep=r"\s+", skiprows=1)

        # Read partition file and create DataFrame with image filenames
        partitions = pd.read_csv(
            partition_file, sep=r"\s+", names=["file", "partition"]
        )

        # Merge attributes with filenames
        # Reset index of data_df to match partitions index
        self.df.reset_index(drop=True, inplace=True)
        # Merge partitions file column with attributes dataframe
        self.df = pd.concat([partitions["file"], self.df], axis=1)

        # Filter by split
        split_map = {"train": 0, "valid": 1, "test": 2}
        self.df = self.df[
            self.df["file"].isin(
                partitions[partitions["partition"] == split_map[self.split]]["file"]
            )
        ]

        # Convert -1/1 labels to 0/1 and ensure they're float values between 0 and 1
        attr_columns = self.df.columns[1:]  # Skip the 'file' column
        self.df[attr_columns] = (self.df[attr_columns] + 1) / 2

        # Extract target and sensitive attributes and convert to float tensors
        self.target = torch.tensor(
            self.df[self.target_attribute].values, dtype=torch.float32
        )
        self.sensitive = torch.tensor(
            self.df[self.sensitive_attribute].values, dtype=torch.float32
        )
