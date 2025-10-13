import os

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainingArguments,
)

from peft import LoraConfig


class SetUp:
    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.data_type = self.config.data_type
        self.revision = self.config.revision
        self.num_cpus = os.cpu_count()
        self.num_fit_workers = min(
            self.num_cpus,
            (config.devices * config.workers_ratio),
        )
        self.num_workers = (
            self.num_cpus if config.use_all_workers else self.num_fit_workers
        )

    def get_dataset(self) -> object:
        dataset: object = instantiate(
            self.config.dataset[self.data_type],
        )
        return dataset()

    def get_model(self) -> SentenceTransformer:
        pretrained_model_name = self.config.pretrained_model_name

        model = SentenceTransformer(
            model_name_or_path=pretrained_model_name,
            revision=self.revision,
        )

        if self.config.is_peft:
            peft_config = LoraConfig(**self.config.peft_config)
            model.add_adapter(peft_config)

        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=self.config.gradient_checkpointing_kwargs,
            )

        return model

    def get_training_arguments(self) -> SentenceTransformerTrainingArguments:
        training_arguments: SentenceTransformerTrainingArguments = instantiate(
            self.config.training_arguments,
            dataloader_num_workers=self.num_workers,
        )
        return training_arguments

    def get_ds_config(self) -> DictConfig:
        if self.config.strategy == "deepspeed":
            ds_config = OmegaConf.to_container(
                self.config.deepspeed,
                resolve=True,
            )
            return ds_config
        return None
