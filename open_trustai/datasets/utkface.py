"""Implementation of UTKFace dataset for fairness research."""

import os
import tarfile
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

# Add import error handling
try:
    import torch
    import gdown
    from PIL import Image
    from torchvision import transforms

    _HAS_VISION_DEPS = True
except ImportError:
    _HAS_VISION_DEPS = False

from .base import VisionDataset


class UTKFaceDataset(VisionDataset):
    """UTKFace Dataset.

    The UTKFace dataset is a large-scale face dataset with long age span.
    The dataset contains over 20,000 face images with annotations of age,
    gender, and ethnicity.

    Sensitive attributes include 'gender', 'race', and 'age'.

    Reference:
    Zhang, Z., Song, Y., & Qi, H. (2017). Age progression/regression by conditional
    adversarial autoencoder. In IEEE Conference on Computer Vision and Pattern
    Recognition (CVPR).
    """

    _FILES = {
        "part1.tar.gz": {
            "id": "1mb5Z24TsnKI3ygNIlX6ZFiwUj0_PmpAW",
            "hash": "md5:4c987669d98b4385d5279056cecdd88b",
        },
        "part2.tar.gz": {
            "id": "19vdaXVRtkP-nyxz1MYwXiFsh_m_OL72b",
            "hash": "md5:ff9a734ffcab5ae235dc7c5e665900b8",
        },
        "part3.tar.gz": {
            "id": "1oj9ZWsLV2-k2idoW_nRSrLQLUP3hus3b",
            "hash": "md5:9038f25ba7173fae23ea805c5f3ba1e4",
        },
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        download: bool = True,
        target_attribute: str = "age",
        sensitive_attribute: str = "gender",
        feature_transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        sensitive_transform: Optional[Any] = None,
        image_size: Tuple[int, int] = (224, 224),
        cache_images: bool = True,
    ):
        """Initialize the UTKFace dataset.

        Args:
            root: Root directory for the dataset
            split: Which split to use ('train', 'valid', or 'test')
            download: Whether to download the dataset if not present
            target_attribute: Target attribute to use (defaults to 'age')
            sensitive_attribute: Sensitive attribute to use (defaults to 'gender')
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

        self.base_folder = "utkface"
        self.split = split
        self.download = download
        self.image_size = image_size
        self.cache_images = cache_images
        self._image_cache = {} if cache_images else None

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
        return ["gender", "race", "age"]

    @property
    def target_attribute_names(self) -> List[str]:
        """List of target attribute names."""
        return ["age", "gender", "race"]

    @property
    def protected_groups(self) -> Dict[str, List[Any]]:
        """Get protected group values for each sensitive attribute."""
        return {
            "gender": [1],  # Female
            "race": [1, 2, 3, 4],  # Non-white races
            "age": list(range(40)),  # Under 40
        }

    @property
    def favourable_outcomes(self) -> Dict[str, List[Any]]:
        """Get favorable outcome values for each target attribute."""
        return {
            "gender": [0],  # Male
            "race": [0],  # White
            "age": list(range(40, 117)),  # Over 40
        }

    @property
    def recommended_metrics(self) -> List[str]:
        """Get recommended fairness metrics for this dataset."""
        return ["statistical_parity", "equal_opportunity", "disparate_impact"]

    @property
    def bias_notes(self) -> str:
        """Get notes about known biases in the dataset."""
        return """
        The UTKFace dataset contains several known biases:
        1. Gender bias: Binary gender labels may not represent gender diversity
        2. Racial bias: Simplified racial categories may not capture ethnic diversity
        3. Age bias: Age distribution may not be uniform
        4. Intersectional bias: Certain combinations of attributes may be underrepresented
        5. Collection bias: Data collection method may favor certain demographics
        6. Label bias: Age labels may have estimation errors
        7. Quality bias: Image quality and conditions may vary across demographics
        
        Note: The dataset uses simplified categories for gender and race,
        which may not capture the full spectrum of human diversity.
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

        os.makedirs(os.path.join(self.root, self.base_folder), exist_ok=True)

        for filename, value in self._FILES.items():
            gdown.cached_download(
                url=f"https://drive.google.com/uc?id={value['id']}",
                path=os.path.join(self.root, self.base_folder, filename),
                hash=value["hash"],
                quiet=False,
            )

            # Extract tar files
            with tarfile.open(
                os.path.join(self.root, self.base_folder, filename), "r:gz"
            ) as tar:
                tar.extractall(os.path.join(self.root, self.base_folder))

    def _load_data(self) -> None:
        """Load the dataset into memory."""
        image_dir = os.path.join(self.root, self.base_folder)
        if not os.path.exists(image_dir):
            raise RuntimeError(
                f"Dataset not found at {image_dir}. Please use download=True to download it."
            )

        # Get all jpg files
        image_files = []
        for root, _, files in os.walk(image_dir):
            for f in files:
                if f.endswith(".jpg"):
                    rel_path = os.path.relpath(root, image_dir)
                    if rel_path == ".":
                        image_files.append(f)
                    else:
                        image_files.append(os.path.join(rel_path, f))

        # Parse attributes from filenames
        data = []
        for filename in image_files:
            attrs = self._parse_filename(filename)
            if attrs is not None:
                data.append({"file": filename, **attrs})

        # Create DataFrame
        self.df = pd.DataFrame(data)

        # Split dataset
        if self.split != "train":
            # Use 80% for training, 10% for validation, 10% for testing
            train_size = int(0.8 * len(self.df))
            val_size = int(0.1 * len(self.df))

            if self.split == "valid":
                self.df = self.df.iloc[train_size : train_size + val_size]
            elif self.split == "test":
                self.df = self.df.iloc[train_size + val_size :]
        else:
            self.df = self.df.iloc[: int(0.8 * len(self.df))]

        # Extract target and sensitive attributes
        # Binarize sensitive attributes based on selection
        sensitive_data = self.df[self.sensitive_attribute].copy()
        if self.sensitive_attribute == "gender":
            # Binarize gender: 1 if female, 0 if male
            sensitive_data = (sensitive_data == 1).astype(int)
        elif self.sensitive_attribute == "race":
            # Binarize race: 1 if white, 0 otherwise
            sensitive_data = (sensitive_data == 0).astype(int)
        elif self.sensitive_attribute == "age":
            # Binarize age: 1 if age >= 30, 0 otherwise
            sensitive_data = (sensitive_data >= 30).astype(int)

        self.df[self.sensitive_attribute] = sensitive_data
        self.target = torch.tensor(self.df[self.target_attribute].values)
        self.sensitive = torch.tensor(self.df[self.sensitive_attribute].values)

    def _parse_filename(self, filename: str) -> Optional[Dict[str, int]]:
        """Parse attributes from filename.

        UTKFace filenames are in the format: [age]_[gender]_[race]_[date&time].jpg
        - age: 0-116
        - gender: 0 (male), 1 (female)
        - race: 0 (White), 1 (Black), 2 (Asian), 3 (Indian), 4 (Others)
        """
        try:
            # Handle potential subdirectory in path
            filename = os.path.basename(filename)
            age, gender, race, _ = filename.split("_")
            return {"age": int(age), "gender": int(gender), "race": int(race)}
        except:
            return None
