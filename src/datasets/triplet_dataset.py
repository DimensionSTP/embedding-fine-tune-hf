from typing import Dict
import os

import importlib

datasets = importlib.import_module("datasets")
HFDataset = datasets.Dataset
load_dataset = datasets.load_dataset


class StructuralDataset:
    def __init__(
        self,
        data_path: str,
        split_ratio: float,
        is_strict_split: bool,
        seed: int,
        dataset_name: str,
        dataset_format: str,
        anchor_column_name: str,
        positive_column_name: str,
        negative_column_name: str,
    ) -> None:
        self.data_path = data_path
        self.split_ratio = split_ratio
        self.is_strict_split = is_strict_split
        self.seed = seed
        self.dataset_name = dataset_name
        self.dataset_format = dataset_format
        self.anchor_column_name = anchor_column_name
        self.positive_column_name = positive_column_name
        self.negative_column_name = negative_column_name

    def __call__(self) -> Dict[str, HFDataset]:
        file_name = f"{self.dataset_name}.{self.dataset_format}"
        full_data_path = os.path.join(
            self.data_path,
            file_name,
        )

        dataset = load_dataset(
            self.dataset_format,
            data_files=full_data_path,
        )["train"]

        output_column_names = [
            self.anchor_column_name,
            self.positive_column_name,
            self.negative_column_name,
        ]
        remove_columns = [
            name for name in dataset.column_names if name not in output_column_names
        ]

        dataset = dataset.remove_columns(remove_columns)

        split_dataset = dataset.train_test_split(
            test_size=self.split_ratio,
            seed=self.seed,
        )

        if self.is_strict_split:
            train_dataset = split_dataset["train"]
        else:
            train_dataset = dataset

        val_dataset = split_dataset["test"]

        return {
            "train": train_dataset,
            "val": val_dataset,
        }
