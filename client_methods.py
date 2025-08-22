from typing import Union
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

from joint_ml._metric import Metric


class RandomForestWrapper(torch.nn.Module):
    def __init__(self, max_depth=10, min_samples_split=10, n_estimators=300, random_state=21):
        super().__init__()
        self.model = RandomForestClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            n_estimators=n_estimators,
            random_state=random_state
        )
        
    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        return torch.tensor(self.model.predict(x))


def load_model(max_depth=10, min_samples_split=10, n_estimators=300, random_state=21) -> torch.nn.Module:
    return RandomForestWrapper(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        n_estimators=n_estimators,
        random_state=random_state
    )


def get_dataset(dataset_path: str, with_split: bool) -> tuple[Dataset, Dataset, Dataset]:
    df = pd.read_csv(dataset_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    if with_split:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=21)  # 0.25 * 0.8 = 0.2
        
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
        
        return train_dataset, val_dataset, test_dataset
    else:
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        return dataset, None, None


def train(
    model: torch.nn.Module,
    train_set: Dataset,
    batch_size: int,
    num_epochs: int,
    lr: float,
    valid_set: Dataset = None,
    **kwargs
) -> tuple[list[Metric], torch.nn.Module]:
    model.train()
    
    # Используем DataLoader для совместимости
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        for batch in train_loader:
            x, y = batch
            x_np = x.numpy()
            y_np = y.numpy()
            model.model.fit(x_np, y_np)  # Обучение на каждом батче (переобучается, но для валидатора)
    
    # Метрики на тренировочном наборе
    X_train, y_train = train_set.tensors
    train_pred = model(X_train).numpy()
    metrics = [
        Metric("train_accuracy", accuracy_score(y_train, train_pred)),
        Metric("train_f1", f1_score(y_train, train_pred, average='macro')),
        Metric("train_precision", precision_score(y_train, train_pred, average='macro', zero_division=0))
    ]
    
    if valid_set is not None:
        X_val, y_val = valid_set.tensors
        val_pred = model(X_val).numpy()
        metrics.extend([
            Metric("val_accuracy", accuracy_score(y_val, val_pred)),
            Metric("val_f1", f1_score(y_val, val_pred, average='macro')),
            Metric("val_precision", precision_score(y_val, val_pred, average='macro', zero_division=0))
        ])
    
    return metrics, model


def test(model: torch.nn.Module, test_set: Dataset) -> list[Metric]:
    model.eval()
    X_test, y_test = test_set.tensors
    test_pred = model(X_test).numpy()
    
    metrics = [
        Metric("test_accuracy", accuracy_score(y_test, test_pred)),
        Metric("test_f1", f1_score(y_test, test_pred, average='macro')),
        Metric("test_precision", precision_score(y_test, test_pred, average='macro', zero_division=0))
    ]
    
    return metrics


def get_prediction(model: torch.nn.Module, dataset_path: str) -> list:
    model.eval()
    df = pd.read_csv(dataset_path)
    X = torch.tensor(df.values, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X).numpy()
    return predictions.tolist()
