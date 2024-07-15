import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np


def save_checkpoint(report: dict, optimizer: torch.optim.Adam, model: torch.nn.Module, path: str):
    """Save parameters of trained model

    :param report: dict:
    :param optimizer: torch.optim.optimizer:
    :param model: torch.nn.Module:
    :param path: str:

    """
    checkpoint = {
        'accuracy': report["accuracy"],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at {path}")


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Adam = None, device: str = 'cpu'):
    """Load parameters of trained model

    :param path: str:
    :param model: torch.nn.Module:
    :param optimizer: torch.optim.optimizer:
    :return: model, optimizer, accuracy
    """
    checkpoint = torch.load(path, map_location=torch.device(device))

    model.load_state_dict(checkpoint['model_state_dict'])
    accuracy = checkpoint['accuracy']
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, accuracy


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute metrics f1, accuracy, precision, recall

    :param y_true: np.ndarray:
    :param y_pred: np.ndarray:
    :param threshold: float:  (Default value = 0.5)

    """
    preds = y_pred >= threshold
    f1_macro = round(f1_score(y_true, preds, average='macro'), 4)
    f1_micro = round(f1_score(y_true, preds, average='micro'), 4)
    f1 = round(f1_score(y_true, preds), 4)

    acc = round(accuracy_score(y_true, preds), 4)
    pr = round(precision_score(y_true, preds), 4)
    re = round(recall_score(y_true, preds), 4)
    roc_auc = round(roc_auc_score(y_true, y_pred), 4)

    return {
        'accuracy': acc,
        'f1': f1,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'precision': pr,
        'recall': re,
        'roc_auc': roc_auc
    }
