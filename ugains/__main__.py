import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything, _logger

from .utils.trainer_utils import (
    debugging,
    finish,
    get_logger,
    load_checkpoint_with_missing_or_exsessive_keys,
    log_hyperparameters,
)

# https://github.com/facebookresearch/hydra/issues/1012
_logger.handlers = []
_logger.propagate = True
torch.set_float32_matmul_precision("medium")
log = get_logger(__name__)
load_dotenv(override=True)


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def train(cfg: DictConfig) -> None:
    debugging(cfg)
    cfg.seed = seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    model = hydra.utils.instantiate(
        cfg.model,
        optimzer_config=cfg.optimizer,
        scheduler_config=cfg.scheduler,
        loss_config=cfg.loss,
        metrics_config=cfg.metrics,
    )
    if cfg.checkpoint is not None:
        log.info(f"Loading checkpoint <{cfg.checkpoint}>")
        state_dict = torch.load(cfg.checkpoint)
        # wrapped in logger to return missing or unexpected keys
        log.warning(
            load_checkpoint_with_missing_or_exsessive_keys(
                model,
                state_dict["state_dict"],
                retrain_transofmer=False,
            )
        )

    callbacks = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    logger = []
    if "logger" in cfg:
        for _, lg_conf in cfg.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from cfg to all lightning loggers
    log.info("Logging hyperparameters!")
    log_hyperparameters(
        config=cfg,
        model=model,
        trainer=trainer,
    )

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set, using the best model achieved during training
    if cfg.get("test_after_training") and not cfg.trainer.get("fast_dev_run"):
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule, ckpt_path="last")

    # Make sure everything closed properly
    log.info("Finalizing!")
    finish(
        trainer=trainer,
    )
    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")
    # Return metric score for hyperparameter optimization
    optimized_metric = cfg.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def validate(cfg: DictConfig) -> None:
    debugging(cfg)
    cfg.seed = seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    model = hydra.utils.instantiate(
        cfg.model,
        optimzer_config=cfg.optimizer,
        scheduler_config=cfg.scheduler,
        loss_config=cfg.loss,
        metrics_config=cfg.metrics,
    )
    if cfg.checkpoint is None:
        log.warning("No checkpoint found!")
    else:
        log.info(f"Loading checkpoint <{cfg.checkpoint}>")
        state_dict = torch.load(cfg.checkpoint)
        # wrapped in logger to return missing or unexpected keys
        log.warning(
            load_checkpoint_with_missing_or_exsessive_keys(
                model, state_dict["state_dict"]
            )
        )

    callbacks = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    logger = []
    if "logger" in cfg:
        for _, lg_conf in cfg.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )
    # Send some parameters from cfg to all lightning loggers
    log.info("Logging hyperparameters!")
    log_hyperparameters(
        config=cfg,
        model=model,
        trainer=trainer,
    )

    # Train the model
    log.info("Starting training!")
    trainer.validate(model=model, datamodule=datamodule)


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def test(cfg: DictConfig) -> None:
    # debugging(cfg)
    cfg.seed = seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    model = hydra.utils.instantiate(
        cfg.model,
        optimzer_config=cfg.optimizer,
        scheduler_config=cfg.scheduler,
        loss_config=cfg.loss,
        metrics_config=cfg.metrics,
    )

    if cfg.checkpoint is None:
        log.warning("No checkpoint found!")
    else:
        log.info(f"Loading checkpoint <{cfg.checkpoint}>")
        state_dict = torch.load(cfg.checkpoint)
        # wrapped in logger to return missing or unexpected keys
        log.warning(
            load_checkpoint_with_missing_or_exsessive_keys(
                model, state_dict["state_dict"]
            )
        )

    callbacks = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    logger = []
    if "logger" in cfg:
        for _, lg_conf in cfg.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )
    # Send some parameters from cfg to all lightning loggers
    log.info("Logging hyperparameters!")
    log_hyperparameters(
        config=cfg,
        model=model,
        trainer=trainer,
    )

    # Train the model
    log.info("Starting training!")
    trainer.test(model=model, datamodule=datamodule)


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def valtest(cfg: DictConfig) -> None:
    debugging(cfg)
    cfg.seed = seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    model = hydra.utils.instantiate(
        cfg.model,
        optimzer_config=cfg.optimizer,
        scheduler_config=cfg.scheduler,
        loss_config=cfg.loss,
        metrics_config=cfg.metrics,
    )
    if cfg.checkpoint is None:
        log.warning("No checkpoint found!")
    else:
        log.info(f"Loading checkpoint <{cfg.checkpoint}>")
        state_dict = torch.load(cfg.checkpoint)
        # wrapped in logger to return missing or unexpected keys
        log.warning(
            load_checkpoint_with_missing_or_exsessive_keys(
                model, state_dict["state_dict"]
            )
        )

    callbacks = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    logger = []
    if "logger" in cfg:
        for _, lg_conf in cfg.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )
    # Send some parameters from cfg to all lightning loggers
    log.info("Logging hyperparameters!")
    log_hyperparameters(
        config=cfg,
        model=model,
        trainer=trainer,
    )

    # Train the model
    log.info("Starting training!")
    trainer.validate(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)
