from typing import Callable

import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import compute_metrics, save_checkpoint


def loss_fn(outputs: np.ndarray, targets: np.ndarray):
    """Loss function (added weights having class disbalance)

    :param outputs: np.ndarray:
    :param targets: np.ndarray:

    """
    return nn.BCELoss()(outputs, targets)


def train_epoch(model: nn.Module, optimizer: torch.optim.Adam, loader: DataLoader, device: str,
                loss_fn: Callable):
    """One train epoch

        :param model: nn.Module:
        :param optimizer: torch.optim.optimizer:
        :param loader: DataLoader:
        :param device: str:
        :param loss_fn: Callable:

        """
    model.train()
    fin_targets = []
    fin_outputs = []
    for _, data in tqdm(enumerate(loader, 0), total=len(loader)):
        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['labels'].to(device, dtype=torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs.flatten(), targets)
        loss.backward()
        optimizer.step()

        fin_targets.extend(targets.cpu().detach().numpy().tolist())
        fin_outputs.extend(outputs.flatten().cpu().detach().numpy().tolist())

    loss = loss_fn(torch.tensor(fin_outputs), torch.tensor(fin_targets))
    print(f'  Train Loss:  {loss.item()}')


def validation_epoch(model: nn.Module, loader: DataLoader, device: str, loss_fn: Callable):
    """One validation epoch

    :param model: nn.Module:
    :param loader: DataLoader:
    :param device: str:
    :param loss_fn: Callable:

    """
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['labels'].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids).flatten()
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
        loss = loss_fn(torch.tensor(fin_outputs), torch.tensor(fin_targets))
        print(f'  Val Loss:  {loss.item()}')

    return fin_targets, fin_outputs


def fit(model: nn.Module, optimizer, device, training_loader: DataLoader, val_loader: DataLoader, epochs: int = 20):
    best_accuracy = 0
    for epoch in range(epochs):
        print(f"EPOCH {epoch}")
        train_epoch(model, optimizer, training_loader, device, loss_fn)
        fin_targets, fin_outputs = validation_epoch(model, val_loader, device, loss_fn)
        report = compute_metrics(np.array(fin_targets), np.array(fin_outputs))
        print(report)

        if best_accuracy < report["accuracy"]:
            save_checkpoint(report, optimizer, model, "best.ckpt")
            best_accuracy = report["accuracy"]
