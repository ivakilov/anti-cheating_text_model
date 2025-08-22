from typing import Union
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score
import torch
from torch.utils.data import Dataset, TensorDataset

from joint_ml import Metric


GET_DATASET_RETURN_TYPE = Union[
    tuple[Dataset, Dataset, Dataset],
    tuple[Dataset, Dataset],
    tuple[Dataset],
]


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
    """
    Метод для генерации модели. На вход будут подаваться параметры, указанные на сервисе как Model Parameters.

    :param max_depth: Максимальная глубина дерева
    :param min_samples_split: Минимальное количество samples для разделения узла
    :param n_estimators: Количество деревьев в лесу
    :param random_state: Seed для воспроизводимости
    :return: Модель RandomForest
    """
    return RandomForestWrapper(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        n_estimators=n_estimators,
        random_state=random_state
    )


def get_dataset(dataset_path: str, with_split: bool, test_size=0.2, val_size=0.2, random_state=21) -> GET_DATASET_RETURN_TYPE:
    """
    Метод для чтения, предобработки и разбития датасета.

    :param dataset_path: путь до csv-файла с датасетом
    :param with_split: параметр разделения набор данных
    :param test_size: доля тестовой выборки
    :param val_size: доля валидационной выборки (от оставшихся после теста данных)
    :param random_state: Seed для воспроизводимости
    :return: Разбитые или неразбитые наборы данных
    """
    df = pd.read_csv(dataset_path)
    
    # Предполагаем, что последний столбец - целевая переменная
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    if with_split:
        # Сначала разделяем на train+val и test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Затем разделяем train+val на train и validation
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio, random_state=random_state
        )
        
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )
        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        )
        
        return train_dataset, val_dataset, test_dataset
    else:
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        )
        return (dataset,)


def train(model: torch.nn.Module, train_set: Dataset, valid_set: Dataset = None) -> tuple[list[Metric], torch.nn.Module]:
    """
    Метод для тренировки модели.
    
    :param model: модель для обучения
    :param train_set: тренировочная выборка
    :param valid_set: валидационная выборка (опционально)
    :return: метрики и обученная модель
    """
    # Извлекаем данные из Dataset
    X_train, y_train = zip(*[(x, y) for x, y in train_set])
    X_train = torch.stack(X_train).numpy()
    y_train = torch.stack(y_train).numpy()
    
    # Обучаем модель
    model.model.fit(X_train, y_train)
    
    # Вычисляем метрики на тренировочной выборке
    train_pred = model.model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    train_f1 = f1_score(y_train, train_pred, average='macro')
    train_precision = precision_score(y_train, train_pred, average='binary')
    
    metrics = [
        Metric("train_accuracy", train_accuracy),
        Metric("train_f1", train_f1),
        Metric("train_precision", train_precision)
    ]
    
    # Если есть валидационная выборка
    if valid_set is not None:
        X_val, y_val = zip(*[(x, y) for x, y in valid_set])
        X_val = torch.stack(X_val).numpy()
        y_val = torch.stack(y_val).numpy()
        
        val_pred = model.model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average='macro')
        val_precision = precision_score(y_val, val_pred, average='binary')
        
        metrics.extend([
            Metric("val_accuracy", val_accuracy),
            Metric("val_f1", val_f1),
            Metric("val_precision", val_precision)
        ])
    
    return metrics, model


def test(model: torch.nn.Module, test_set: Dataset, return_output: bool = False) -> tuple[list[Metric]] | tuple[list[Metric], list]:
    """
    Метод для тестировки модели на данных.
    
    :param model: модель для тестирования
    :param test_set: тестовая выборка
    :param return_output: возврат результатов прогноза
    :return: метрики и (опционально) предсказания
    """
    X_test, y_test = zip(*[(x, y) for x, y in test_set])
    X_test = torch.stack(X_test).numpy()
    y_test = torch.stack(y_test).numpy()
    
    test_pred = model.model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='macro')
    test_precision = precision_score(y_test, test_pred, average='binary')
    
    metrics = [
        Metric("test_accuracy", test_accuracy),
        Metric("test_f1", test_f1),
        Metric("test_precision", test_precision)
    ]
    
    if return_output:
        return metrics, test_pred.tolist()
    return (metrics,)


def get_prediction(model: torch.nn.Module, dataset_path: str) -> list:
    """
    Метод для получения предсказаний модели на данных пользователя.
    
    :param model: обученная модель
    :param dataset_path: путь до данных для предсказания
    :return: список предсказаний
    """
    df = pd.read_csv(dataset_path)
    X = df.values
    predictions = model.model.predict(X)
    return predictions.tolist()
