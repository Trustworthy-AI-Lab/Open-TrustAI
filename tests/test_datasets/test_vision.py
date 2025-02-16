import os
import pytest
import numpy as np

# Check for vision dependencies
vision_available = True
try:
    import torch
    from PIL import Image
    import torchvision
    from torchvision import transforms
    import gdown
except ImportError:
    vision_available = False

from open_trustai.datasets.celeba import CelebADataset
from open_trustai.datasets.utkface import UTKFaceDataset
from open_trustai.datasets.fairface import FairFaceDataset

# Skip all tests if vision dependencies not installed
pytestmark = pytest.mark.skipif(
    not vision_available,
    reason="Vision dependencies not installed. Run: pip install torch torchvision Pillow gdown",
)


# Common test mixins
class ImageDatasetTestMixin:
    """Common tests for all image datasets"""

    @pytest.fixture
    def dataset(self):
        """Should be implemented by child classes to return dataset instance"""
        raise NotImplementedError

    def test_basic_properties(self, dataset):
        """Test common properties that all image datasets should have"""
        # Test required properties
        assert isinstance(dataset.feature_names, list)
        assert isinstance(dataset.sensitive_attribute_names, list)
        assert isinstance(dataset.target_attribute_names, list)
        assert isinstance(dataset.protected_groups, dict)
        assert isinstance(dataset.favourable_outcomes, dict)
        assert isinstance(dataset.recommended_metrics, list)
        assert isinstance(dataset.bias_notes, str)

    def test_image_properties(self, dataset):
        """Test image tensor properties"""
        img, target, sensitive = dataset[0]

        assert isinstance(img, torch.Tensor)
        assert img.dtype == torch.float32
        assert len(img.shape) == 3  # C,H,W format
        assert img.shape[0] == 3  # RGB channels

    def test_custom_transforms(self, dataset, tmp_path):
        """Test applying custom transforms"""
        feature_transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]
        )

        dataset.feature_transform = feature_transform
        img, _, _ = dataset[0]
        assert img.shape == (3, 32, 32)


class TestCelebADataset(ImageDatasetTestMixin):
    @pytest.fixture(scope="session")
    def celeba_root(self, tmp_path_factory):
        """Download a small subset of CelebA for testing"""
        root = tmp_path_factory.mktemp("celeba")
        try:
            CelebADataset(
                root=str(root),
                split="train",
                download=True,
                image_size=(64, 64),
            )
        except Exception as e:
            pytest.skip(f"Failed to download CelebA dataset: {str(e)}")
        return root

    @pytest.fixture
    def dataset(self, celeba_root):
        return CelebADataset(
            root=str(celeba_root), split="train", download=False, image_size=(64, 64)
        )

    def test_splits(self, celeba_root):
        """Test all dataset splits"""
        for split in ["train", "valid", "test"]:
            dataset = CelebADataset(
                root=str(celeba_root), split=split, download=False, image_size=(64, 64)
            )
            assert len(dataset) > 0

    def test_attributes(self, celeba_root):
        """Test different attribute combinations"""
        dataset = CelebADataset(
            root=str(celeba_root),
            split="train",
            target_attribute="Young",
            sensitive_attribute="Male",
            download=False,
            image_size=(64, 64),
        )

        assert dataset.target_attribute == "Young"
        assert dataset.sensitive_attribute == "Male"

        img, target, sensitive = dataset[0]
        assert target in [0, 1]
        assert sensitive in [0, 1]


class TestUTKFaceDataset(ImageDatasetTestMixin):
    @pytest.fixture(scope="session")
    def utkface_root(self, tmp_path_factory):
        """Download a small subset of UTKFace for testing"""
        root = tmp_path_factory.mktemp("utkface")
        try:
            UTKFaceDataset(root=str(root), download=True, image_size=(64, 64))
        except Exception as e:
            pytest.skip(f"Failed to download UTKFace dataset: {str(e)}")
        return root

    @pytest.fixture
    def dataset(self, utkface_root):
        return UTKFaceDataset(
            root=str(utkface_root), download=False, image_size=(64, 64)
        )

    def test_attributes(self, utkface_root):
        """Test different attribute combinations"""
        dataset = UTKFaceDataset(
            root=str(utkface_root),
            target_attribute="race",
            sensitive_attribute="gender",
            download=False,
            image_size=(64, 64),
        )

        assert dataset.target_attribute == "race"
        assert dataset.sensitive_attribute == "gender"

        img, target, sensitive = dataset[0]
        assert target in range(5)  # race has 5 categories
        assert sensitive in [0, 1]  # gender is binary

    def test_cache(self, utkface_root):
        """Test image caching functionality"""
        dataset = UTKFaceDataset(
            root=str(utkface_root),
            download=False,
            cache_images=True,
            image_size=(64, 64),
        )

        # Access same image twice to test cache
        img1, _, _ = dataset[0]
        img2, _, _ = dataset[0]
        assert torch.equal(img1, img2)
