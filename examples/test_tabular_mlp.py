from open_trustai.datasets import (
    BankMarketingDataset,
    AdultDataset,
    GermanCreditDataset,
    CompasDataset,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Type, Dict, Any, Tuple, List, Optional
import pandas as pd


class SimpleModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DatasetManager:
    """Manages different datasets and their configurations."""

    DATASETS = {
        "bank": BankMarketingDataset,
        "adult": AdultDataset,
        "german": GermanCreditDataset,
        "compas": CompasDataset,
    }

    @staticmethod
    def get_dataset(name: str, **kwargs) -> Any:
        """
        Get a dataset by name with appropriate configuration.

        Args:
            name: Name of the dataset
            **kwargs: Additional arguments to pass to dataset constructor

        Returns:
            Initialized dataset instance

        Raises:
            ValueError: If dataset name is not found
        """
        if name not in DatasetManager.DATASETS:
            raise ValueError(
                f"Dataset {name} not found. Available datasets: {list(DatasetManager.DATASETS.keys())}"
            )

        dataset_class = DatasetManager.DATASETS[name]
        base_kwargs = DatasetManager._get_base_kwargs(name)
        base_kwargs.update(kwargs)
        return dataset_class(**base_kwargs)

    @staticmethod
    def _get_base_kwargs(name: str) -> Dict[str, Any]:
        """Get default kwargs for a given dataset."""
        base_kwargs = {"root": "data", "download": True}

        if name == "bank":
            base_kwargs["split"] = "full"
        elif name == "adult":
            base_kwargs["split"] = "train"

        return base_kwargs

    @staticmethod
    def get_dataset_info(dataset) -> Dict[str, Any]:
        """
        Get basic information about the dataset.

        Returns:
            Dictionary containing dataset information
        """
        return {
            "type": dataset.__class__.__name__,
            "total_samples": len(dataset),
            "num_features": len(dataset.feature_names),
            "categorical_features": dataset.categorical_features,
            "numerical_features": dataset.numerical_features,
        }

    @staticmethod
    def get_fairness_info(dataset) -> Dict[str, Any]:
        """
        Get fairness-related information about the dataset.

        Returns:
            Dictionary containing fairness information
        """
        return {
            "sensitive_attributes": dataset.sensitive_attribute_names,
            "protected_groups": dataset.protected_groups,
            "target_attributes": dataset.target_attribute_names,
            "favourable_outcomes": dataset.favourable_outcomes,
            "recommended_metrics": dataset.recommended_metrics,
            "bias_notes": dataset.bias_notes,
        }


class ModelTrainer:
    """Handles model training and evaluation."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.criterion = criterion or nn.BCELoss()
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=0.001)
        self.device = device

    def train_epoch(self) -> Tuple[float, float]:
        """
        Train the model for one epoch.

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for features, targets, _ in self.train_loader:
            features = features.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(features).squeeze()
            loss = self.criterion(outputs, targets.float())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

        return total_loss / len(self.train_loader), 100 * correct / total

    def train(self, num_epochs: int = 5) -> List[Tuple[float, float]]:
        """
        Train the model for specified number of epochs.

        Args:
            num_epochs: Number of epochs to train

        Returns:
            List of (loss, accuracy) tuples for each epoch
        """
        metrics = []
        for epoch in range(num_epochs):
            avg_loss, accuracy = self.train_epoch()
            metrics.append((avg_loss, accuracy))

        return metrics


def run_dataset_experiment(
    dataset_name: str, num_epochs: int = 3, batch_size: int = 32
) -> Dict[str, Any]:
    """
    Run a complete experiment for a given dataset.

    Args:
        dataset_name: Name of the dataset to use
        num_epochs: Number of epochs to train
        batch_size: Batch size for training

    Returns:
        Dictionary containing experiment results
    """
    dataset = DatasetManager.get_dataset(dataset_name)
    dataset_info = DatasetManager.get_dataset_info(dataset)
    fairness_info = DatasetManager.get_fairness_info(dataset)

    # Export dataset to CSV before training
    features = pd.DataFrame(dataset.features.numpy())
    sensitive = pd.DataFrame(dataset.sensitive.numpy())
    target = pd.DataFrame(dataset.target.numpy())

    # Save to CSV
    output_file = f"{dataset_name}_dataset.csv"
    pd.concat([features, sensitive, target], axis=1).to_csv(output_file, index=False)
    print(f"Dataset exported to {output_file}")

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SimpleModel(len(dataset.feature_names))
    trainer = ModelTrainer(model, train_loader)
    training_metrics = trainer.train(num_epochs=num_epochs)

    return {
        "dataset_info": dataset_info,
        "fairness_info": fairness_info,
        "training_metrics": training_metrics,
    }


def main():
    datasets_to_test = ["bank", "adult", "german", "compas"]

    for dataset_name in datasets_to_test:
        print(f"\n{'='*70}")
        print(f"Testing {dataset_name.upper()} dataset")
        print(f"{'='*70}")

        try:
            results = run_dataset_experiment(dataset_name)

            # Print results
            print("\nDataset Information:")
            print("-" * 50)
            for key, value in results["dataset_info"].items():
                print(f"{key}: {value}")

            print("\nFairness Information:")
            print("-" * 50)
            for key, value in results["fairness_info"].items():
                print(f"{key}: {value}")

            print("\nTraining Results:")
            print("-" * 50)
            for epoch, (loss, accuracy) in enumerate(results["training_metrics"], 1):
                print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.2f}%")

        except Exception as e:
            print(f"Error processing {dataset_name} dataset: {str(e)}")


if __name__ == "__main__":
    main()
