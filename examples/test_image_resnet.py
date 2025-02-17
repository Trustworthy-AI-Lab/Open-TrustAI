"""Test script for training fair ResNet models on multiple image datasets."""

from open_trustai.datasets import CelebADataset, FairFaceDataset, UTKFaceDataset
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from typing import Dict, Any, Tuple, List, Optional
from tqdm import tqdm


class CustomResNet(nn.Module):
    """ResNet model adapted for image classification."""

    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super().__init__()
        # Load pretrained ResNet
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)

        # Modify final layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
            nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)


class ImageDatasetManager:
    """Manages different image datasets and their configurations."""

    DATASETS = {
        "celeba": CelebADataset,
        "fairface": FairFaceDataset,
        "utkface": UTKFaceDataset,
    }

    @staticmethod
    def get_dataset(name: str, split: str = "train", **kwargs) -> Any:
        """Get a dataset by name with appropriate configuration."""
        if name not in ImageDatasetManager.DATASETS:
            raise ValueError(
                f"Dataset {name} not found. Available: {list(ImageDatasetManager.DATASETS.keys())}"
            )

        dataset_class = ImageDatasetManager.DATASETS[name]
        base_kwargs = ImageDatasetManager._get_base_kwargs(name, split)
        base_kwargs.update(kwargs)
        return dataset_class(**base_kwargs)

    @staticmethod
    def _get_base_kwargs(name: str, split: str) -> Dict[str, Any]:
        """Get default kwargs for a given dataset."""
        base_kwargs = {"root": f"data", "split": split, "download": True}

        if name == "celeba":
            base_kwargs.update(
                {"target_attribute": "Attractive", "sensitive_attribute": "Male"}
            )
        elif name == "fairface":
            base_kwargs.update(
                {"target_attribute": "gender", "sensitive_attribute": "race"}
            )
        elif name == "utkface":
            base_kwargs.update(
                {"target_attribute": "gender", "sensitive_attribute": "race"}
            )

        return base_kwargs

    @staticmethod
    def get_dataset_info(dataset) -> Dict[str, Any]:
        """Get basic information about the dataset."""
        return {
            "type": dataset.__class__.__name__,
            "total_samples": len(dataset),
            "target_attributes": dataset.target_attribute_names,
            "sensitive_attribute": dataset.sensitive_attribute_names,
            "image_size": dataset[0][0].shape,
        }


class ImageModelTrainer:
    """Handles model training and evaluation for image datasets."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or nn.BCELoss()
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=0.001)
        self.device = device

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        with tqdm(self.train_loader, desc="Training") as pbar:
            for images, targets, _ in pbar:
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images).squeeze()
                loss = self.criterion(outputs, targets.float())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                predicted = (outputs >= 0.5).float()
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

                # Update progress bar
                pbar.set_postfix({"loss": loss.item(), "acc": 100 * correct / total})

        return total_loss / len(self.train_loader), 100 * correct / total

    def evaluate(self) -> Tuple[float, float]:
        """Evaluate the model."""
        if not self.val_loader:
            return 0.0, 0.0

        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, targets, _ in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images).squeeze()
                loss = self.criterion(outputs, targets.float())

                total_loss += loss.item()
                predicted = (outputs >= 0.5).float()
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        return total_loss / len(self.val_loader), 100 * correct / total

    def train(self, num_epochs: int = 5) -> List[Dict[str, float]]:
        """Train the model for specified number of epochs."""
        metrics = []
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.evaluate()

            epoch_metrics = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            metrics.append(epoch_metrics)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        return metrics


def run_dataset_experiment(
    dataset_name: str, num_epochs: int = 3, batch_size: int = 32
) -> Dict[str, Any]:
    """Run a complete experiment for a given dataset."""

    # Get train and validation datasets
    train_dataset = ImageDatasetManager.get_dataset(dataset_name, split="train")
    val_dataset = ImageDatasetManager.get_dataset(dataset_name, split="valid")

    dataset_info = ImageDatasetManager.get_dataset_info(train_dataset)

    # Export dataset to CSV before training
    sensitive = pd.DataFrame(train_dataset.sensitive.numpy())
    target = pd.DataFrame(train_dataset.target.numpy())

    # Save to CSV
    output_file = f"{dataset_name}_dataset.csv"
    pd.concat([sensitive, target], axis=1).to_csv(output_file, index=False)
    print(f"Dataset exported to {output_file}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize model and trainer
    model = CustomResNet(num_classes=1)
    trainer = ImageModelTrainer(model, train_loader, val_loader)

    # Train model
    training_metrics = trainer.train(num_epochs=num_epochs)

    return {
        "dataset_info": dataset_info,
        "training_metrics": training_metrics,
    }


def main():
    datasets_to_test = ["celeba", "fairface", "utkface"]
    # datasets_to_test = ["utkface"]

    for dataset_name in datasets_to_test:
        print(f"\n{'='*70}")
        print(f"Testing {dataset_name.upper()} dataset")
        print(f"{'='*70}")

        try:
            results = run_dataset_experiment(dataset_name)

            print("\nDataset Information:")
            print("-" * 50)
            for key, value in results["dataset_info"].items():
                print(f"{key}: {value}")

            print("\nTraining Results:")
            print("-" * 50)
            for epoch, metrics in enumerate(results["training_metrics"], 1):
                print(f"\nEpoch {epoch}:")
                for metric_name, value in metrics.items():
                    print(f"{metric_name}: {value:.4f}")

        except Exception as e:
            print(f"Error processing {dataset_name} dataset: {str(e)}")


if __name__ == "__main__":
    main()
