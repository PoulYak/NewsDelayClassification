# src/__init__.py

from .data_processing import NewsDataset, NewsDatasetTest, get_test_dataset_dataloader, get_dataset_dataloader, add_augmented_data
from .model_training import train_epoch, validation_epoch, fit, loss_fn
from .model_evaluation import test, load_model, predict
from .utils import save_checkpoint, load_checkpoint, compute_metrics

__all__ = [
    "NewsDataset",
    "NewsDatasetTest",
    "get_test_dataset_dataloader",
    "get_dataset_dataloader",
    "add_augmented_data",
    "train_epoch",
    "validation_epoch",
    "fit",
    "loss_fn",
    "test",
    "load_model",
    "predict",
    "save_checkpoint",
    "load_checkpoint",
    "compute_metrics"
]

__version__ = '1.0.0'