from typing import List, Sequence, Union

import logging
import os
import warnings
from collections.abc import MutableMapping
from pathlib import Path

import git
import pytorch_lightning as pl
import rich.syntax
import rich.tree
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


logger = get_logger(__name__)


def load_checkpoint_with_missing_or_exsessive_keys(
    model, state_dict, retrain_transofmer=False
):
    correct_dict = dict(model.state_dict())

    # if parametrs not found in checkpoint they will be randomly initialized
    for key in state_dict.keys():
        if correct_dict.pop(key, None) is None:
            logger.warning(f"Key not found, it will be initialized randomly: {key}")
    # missing keys
    for key in correct_dict.keys():
        logger.warning(f"Key not found, it will be initialized randomly: {key}")

    # if parametrs have different shape, it will randomly initialize
    correct_dict = dict(model.state_dict())
    for key in correct_dict.keys():
        if (
            (key in state_dict.keys())
            and (key in correct_dict.keys())
            and (state_dict[key].shape != correct_dict[key].shape)
        ):
            logger.warning(
                f"incorrect shape {key}:{state_dict[key].shape} vs {correct_dict[key].shape}"
            )
            logger.warning("HARDCODED HALF-INITIALIZATION")
            if "query" in key:
                correct_dict[key][: state_dict[key].shape[0], ...] = state_dict[key]
                state_dict.update({key: correct_dict[key]})
            else:
                state_dict.update({key: correct_dict[key]})

    # if we have more keys just discard them
    correct_dict = dict(model.state_dict())
    new_state_dict = dict(model.state_dict())
    for key in state_dict.keys():
        if key in correct_dict.keys():
            if retrain_transofmer and ("sem_seg_head" in key):
                logger.warning(f"Skipping: {key}")
                continue
            new_state_dict.update({key: state_dict[key]})
        else:
            logger.warning(f"excessive key: {key}")
    logger.info(model.load_state_dict(new_state_dict))


def ignore_warnings():
    log = get_logger()
    log.info("Disabling python warnings! <config.ignore_warnings=True>")
    warnings.filterwarnings("ignore")


def debugging(config: DictConfig) -> None:
    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("experiment") == "debug":
        # enable adding new keys to config
        OmegaConf.set_struct(config, False)

        log = get_logger()
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True
        log.info(
            "Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>"
        )
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

        # disable adding new keys to config
        OmegaConf.set_struct(config, True)
        return
    # if no debugging - check for commit
    print_config(config, print_all=True)
    git.refresh("/usr/bin/git")
    if git.Repo(config.work_dir).is_dirty():
        logger.warning("repo is dirty, commit first")


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "datamodule",
        "callbacks",
        "logger",
        "seed",
    ),
    resolve: bool = True,
    print_all: bool = False,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    if print_all:
        fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    trainer: pl.Trainer,
    ignore: List[Union[str, None]] = [
        None,
    ],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.
    Additionaly saves:
        - number of trainable model parameters
    """

    # choose which parts of hydra config will be saved to loggers
    ignore = list(set(ignore))
    hparams = {k: v for k, v in config.items() if k not in ignore}

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    def empty(null):
        pass

    trainer.logger.log_hyperparams = empty


def finish(
    trainer: pl.Trainer,
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in trainer.loggers:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()


def flatten_dict(d, parent_key="", sep="_"):
    """
    https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def freeze_until(net, param_name: str = None):
    """
    Freeze net until param_name
    https://opendatascience.slack.com/archives/CGK4KQBHD/p1588373239292300?thread_ts=1588105223.275700&cid=CGK4KQBHD
    Args:
        net:
        param_name:
    Returns:
    """
    found_name = False
    for name, params in net.named_parameters():
        if name == param_name:
            found_name = True
        params.requires_grad = found_name


class set_directory:
    """Sets the cwd within the context

    Args:
        path (Path): The path to the cwd
    """

    def __init__(self, path: Path):
        self.path = path
        self.origin = Path().absolute()

    def __enter__(self):
        os.chdir(self.path)

    def __exit__(self, type, value, traceback):
        os.chdir(self.origin)
