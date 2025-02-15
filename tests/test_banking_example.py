import pytest
import torch
from examples.banking_example import DatasetManager, SimpleModel, ModelTrainer


def test_dataset_manager():
    # Test dataset loading
    dataset = DatasetManager.get_dataset("bank")
    assert dataset is not None

    # Test invalid dataset name
    with pytest.raises(ValueError):
        DatasetManager.get_dataset("invalid_dataset")

    # Test dataset info
    info = DatasetManager.get_dataset_info(dataset)
    assert isinstance(info, dict)
    assert "total_samples" in info
    assert "num_features" in info


def test_simple_model():
    model = SimpleModel(input_dim=10)
    x = torch.randn(32, 10)
    output = model(x)
    assert output.shape == (32, 1)


def test_model_trainer():
    # Create mock data
    features = torch.randn(100, 10)
    targets = torch.randint(0, 2, (100,))
    mock_dataset = torch.utils.data.TensorDataset(
        features, targets, torch.zeros(100)  # Mock sensitive attributes
    )
    mock_loader = torch.utils.data.DataLoader(mock_dataset, batch_size=32)

    model = SimpleModel(input_dim=10)
    trainer = ModelTrainer(model, mock_loader)

    # Test training
    metrics = trainer.train(num_epochs=2)
    assert len(metrics) == 2
    for loss, accuracy in metrics:
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
