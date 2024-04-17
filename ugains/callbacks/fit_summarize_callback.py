from pytorch_lightning.Callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
import wandb
import numpy as np
from PIL import Image as PILImage
import pandas as pd


class FitSummarizeCallback(Callback):
    def __init__(self):
        super().__init__()

    def teardown(trainer, pl_module):
        # Called when fit, validate, test, predict, or tune ends.
        logger = None
        for logger in pl_module.loggers:
            if isinstance(logger, WandbLogger):
                logger = logger.experiment
        if logger is None:
            return

        logger.log(
            {
                "train_features": pl_module.train_features.get_wandb_figure(
                    embedding=pl_module.train_features.fit().transform(),
                    trainid2color=pl_module.trainer.datamodule.inlier_trainid2color,
                    trainid2name=pl_module.trainer.datamodule.inlier_trainid2name,
                )
            }
        )
        logger.log(
            {
                "train_features_ood": pl_module.ood_features.get_wandb_figure(
                    embedding=pl_module.train_features.transform(
                        pl_module.ood_features.features
                    ),
                    trainid2color=pl_module.trainer.datamodule.inlier_trainid2color,
                    trainid2name=pl_module.trainer.datamodule.inlier_trainid2name,
                )
            }
        )
        target = np.repeat(
            np.arange(pl_module.train_features.num_classes),
            pl_module.train_features.samples_per_class,
        )
        target_ood = np.repeat(
            np.arange(pl_module.ood_features.num_classes),
            pl_module.ood_features.samples_per_class,
        )
        logger.log(
            {
                "ground_features": pl_module.train_features.get_wandb_figure(
                    embedding=pl_module.train_features.fit(
                        pl_module.train_features.features[
                            : pl_module.train_features.samples_per_class
                        ]
                    ).transform(
                        pl_module.train_features.features[
                            : pl_module.train_features.samples_per_class
                        ]
                    ),
                    trainid2color=pl_module.trainer.datamodule.inlier_trainid2color,
                    trainid2name=pl_module.trainer.datamodule.inlier_trainid2name,
                    target=target[: pl_module.train_features.samples_per_class],
                )
            }
        )
        logger.log(
            {
                "ground_features_ood": pl_module.ood_features.get_wandb_figure(
                    embedding=pl_module.train_features.fit(
                        pl_module.train_features.features[
                            : pl_module.train_features.samples_per_class
                        ]
                    ).transform(
                        pl_module.ood_features.features[
                            : pl_module.ood_features.samples_per_class
                        ]
                    ),
                    trainid2color=pl_module.trainer.datamodule.inlier_trainid2color,
                    trainid2name=pl_module.trainer.datamodule.inlier_trainid2name,
                    target=target_ood[: pl_module.ood_features.samples_per_class],
                )
            }
        )

        for idx in range(len(pl_module.train_log_table)):
            uncertainty = pl_module.train_log_table[idx]["uncertainty"]
            uncertainty_image = PILImage.fromarray(
                (
                    (
                        (uncertainty - uncertainty.min())
                        / (uncertainty.max() - uncertainty.min())
                    )
                    * 255
                ).astype(np.uint8),
                mode="L",
            )
            pl_module.train_log_table[idx]["uncertainty"] = wandb.Image(
                uncertainty_image
            )
            pl_module.train_log_table[idx]["image"] = pl_module.normalize_image(
                pl_module.train_log_table[idx]["image"]
            )
            for k, v in pl_module.train_log_table[idx].items():
                if isinstance(v, np.ndarray):
                    image = PILImage.fromarray(v, mode="RGB")
                    pl_module.train_log_table[idx][k] = wandb.Image(image)

        df = pd.DataFrame(pl_module.train_log_table)
        table = wandb.Table(dataframe=df)
        logger.log({"train_predictions": table})
        pl_module.train_log_table = []
