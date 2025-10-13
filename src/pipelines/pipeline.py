import os

from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf

import wandb

from ..utils import SetUp


def train(
    config: DictConfig,
) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        wandb.init(
            project=config.project_name,
            name=config.logging_name,
        )

    setup = SetUp(config)

    train_dataset = setup.get_dataset()["train"]
    val_dataset = setup.get_dataset()["val"]

    model = setup.get_model()

    training_arguments = setup.get_training_arguments()

    ds_config = setup.get_ds_config()
    if ds_config:
        training_arguments.deepspeed = ds_config

    loss_config = OmegaConf.to_container(
        config.loss,
        resolve=True,
    )
    loss_config.pop(
        "_target_",
        None,
    )

    LossClass = get_class(config.loss._target_)

    loss = LossClass(
        model,
        **loss_config,
    )

    trainer_config = OmegaConf.to_container(
        config.trainer,
        resolve=True,
    )
    trainer_config.pop(
        "_target_",
        None,
    )

    TrainerClass = get_class(config.trainer._target_)

    trainer = TrainerClass(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
        **trainer_config,
    )

    try:
        trainer.train(
            resume_from_checkpoint=(
                config.resume_from_checkpoint if config.resume_training else None
            )
        )

        if local_rank == 0:
            wandb.run.alert(
                title="Training Complete",
                text=f"Training process on {config.dataset_name} has successfully finished.",
                level="INFO",
            )
    except Exception as e:
        if local_rank == 0:
            wandb.run.alert(
                title="Training Error",
                text=f"An error occurred during training on {config.dataset_name}: {e}",
                level="ERROR",
            )
        raise e
