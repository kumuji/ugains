from typing import Any

from copy import deepcopy

import numpy as np

import hydra
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from PIL import Image as PILImage
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from ugains.utils.plot_utils import overlay_mask_on_image

from .segment_anything import SamPredictor, sam_model_registry
from .segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from ugains.models.sampler import NLSSampler


class Segmentation(LightningModule):
    def __init__(
        self,
        model_config,
        optimzer_config,
        scheduler_config,
        loss_config,
        metrics_config,
    ):
        super().__init__()
        if metrics_config is not None:
            self.metrics = hydra.utils.instantiate(metrics_config)
        self.optimizer = optimzer_config
        self.scheduler = scheduler_config
        self.inlier_criterion = hydra.utils.instantiate(loss_config)

        self.train_log_table = []
        self.max_log_train = 2
        self.validation_log_table = []
        self.max_log_validation = 2
        self.test_log_table = []
        self.max_log_test = -1

        self.nms_thresh = model_config["nms_thresh"]
        self.automask = model_config["automask"]
        self.minimal_uncert_value_instead_zero = model_config[
            "minimal_uncert_value_instead_zero"
        ]
        self.visualize = model_config["visualize"]

        self.ignore_mask = None
        if model_config["ignore_mask"]:
            self.ignore_mask = torch.load(model_config["ignore_mask"]).cuda()
        self.test_proposal_sampler = NLSSampler(
            model_path=model_config["model_path"],
            thresh=model_config["uncertainty_thresh"],
            method=model_config["sampling_method"],
            ignore_mask=model_config["ignore_mask"],
            num_points=model_config["num_samples"],
        )
        sam_checkpoint = model_config["sam_checkpoint"]
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        self.predictor = SamPredictor(self.sam)
        if self.automask:
            self.automask_generator = SamAutomaticMaskGenerator(
                model=self.sam, min_mask_region_area=10
            )

    def training_step(self, batch, batch_idx=None):
        # get features from frozen encoder
        x, targets = batch

        inlier_targets = []
        ood_targets = []
        inlier_targets, ood_targets = self.split_targets(targets)

        bs = x.shape[0]
        with torch.no_grad():
            features = self.sam.image_encoder(x)

        # get embeddings
        masks = []
        scores = []

        # sample a bunch of points
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
        sparse_embeddings = self.semseg_embedding.weight.unsqueeze(1)
        dense_embeddings = dense_embeddings[:, :, :32, :]

        masks = []
        scores = []
        for i in range(bs):
            low_res_masks, iou_predictions, class_prediction = self.sam.mask_decoder(
                image_embeddings=features[i : i + 1],
                image_pe=self.sam.prompt_encoder.get_dense_pe()[:, :, :32, :],
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            masks.append(low_res_masks.squeeze(1))
            scores.append(class_prediction.squeeze(1))
        masks = torch.stack(masks)
        scores = torch.stack(scores)
        output = {"pred_logits": scores, "pred_masks": masks}

        losses = self.inlier_criterion(output, inlier_targets)
        losses.pop("indices", None)
        for k, v in losses.items():
            losses[k] *= self.inlier_criterion.weight_dict[k]
            self.log(
                f"train/{k}",
                v.detach().cpu(),
                on_epoch=True,
                batch_size=x.shape[0],
            )
        loss = sum(losses.values())
        self.log("train/loss", loss, batch_size=x.shape[0])

        if (len(self.train_log_table) <= self.max_log_train) and (
            self.image_logger is not None
        ):
            seg_maps = []
            for t in targets:
                seg_maps.append(t["semantic"])
            seg_maps = torch.stack(seg_maps)

            semseg = self.semantic_inference(
                output["pred_logits"],
                F.interpolate(
                    output["pred_masks"],
                    size=seg_maps.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ),
            ).max(1)[1]
            for sample_idx in range(x.shape[0]):
                self.train_log_table.append(
                    {
                        "image": x[sample_idx].detach().cpu().numpy(),
                        "inlier_prediction": self.get_prediction_image(
                            x[sample_idx],
                            semseg[sample_idx],
                            self.trainer.datamodule.inlier_trainid2color,
                        ),
                        "inlier_gt": self.get_prediction_image(
                            x[sample_idx],
                            seg_maps[sample_idx],
                            self.trainer.datamodule.inlier_trainid2color,
                        ),
                    }
                )
        return {"loss": loss}

    def on_train_epoch_end(self, outputs: Any):
        if self.image_logger is None:
            return
        self.train_log_table = self.log_image_table(
            self.train_log_table, self.image_logger, mode="train"
        )

    def validation_step(self, batch, batch_idx=None):
        # get features from frozen encoder
        x, target = batch

        bs = x.shape[0]
        with torch.no_grad():
            features = self.sam.image_encoder(x)

        # get embeddings
        masks = []
        scores = []

        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
        sparse_embeddings = self.semseg_embedding.weight.unsqueeze(1)
        dense_embeddings = dense_embeddings[:, :, :32, :]

        masks = []
        scores = []
        for i in range(bs):
            low_res_masks, iou_predictions, class_prediction = self.sam.mask_decoder(
                image_embeddings=features[i : i + 1],
                image_pe=self.sam.prompt_encoder.get_dense_pe()[:, :, :32, :],
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            masks.append(low_res_masks.squeeze(1))
            scores.append(class_prediction.squeeze(1))
        masks = torch.stack(masks)
        scores = torch.stack(scores)
        output = {"pred_logits": scores, "pred_masks": masks}

        seg_maps = []
        for t in target:
            seg_maps.append(t["semantic"])
        seg_maps = torch.stack(seg_maps)

        semseg = self.semantic_inference(
            output["pred_logits"],
            F.interpolate(
                output["pred_masks"],
                size=seg_maps.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ),
        ).max(1)[1]
        self.metrics["iou"].add(semseg.reshape(-1), seg_maps.reshape(-1))

        if len(self.validation_log_table) <= self.max_log_validation:
            for sample_idx in range(x.shape[0]):
                self.validation_log_table.append(
                    {
                        "image": x[sample_idx].detach().cpu().numpy(),
                        "inlier_prediction": self.get_prediction_image(
                            x[sample_idx],
                            semseg[sample_idx],
                            self.trainer.datamodule.inlier_trainid2color,
                        ),
                        "inlier_gt": self.get_prediction_image(
                            x[sample_idx],
                            seg_maps[sample_idx],
                            self.trainer.datamodule.inlier_trainid2color,
                        ),
                    }
                )
        return dict()

    def on_validation_epoch_end(self):
        results = dict()
        iou = self.metrics["iou"].value()["IoU"]
        for i in range(len(iou)):
            metric_name = self.trainer.datamodule.inlier_trainid2name[i]
            results["val_iou_" + metric_name] = iou[i]
        results["val_miou"] = np.nanmean(iou)
        self.metrics["iou"].reset()
        self.log_dict(results)

        if self.image_logger is None:
            return
        self.validation_log_table = self.log_image_table(
            self.validation_log_table, self.image_logger, mode="val"
        )

    def forward(self, x, query_coordinates, instance_target, uncertainty):
        self.predictor.set_torch_image(x, original_image_size=(1024, 2048))

        def apply_coords(coords):
            old_h, old_w = 1024, 2048
            new_h, new_w = 512, 1024
            coords = deepcopy(coords).float()
            coords[..., 0] = coords[..., 0] * (new_w / old_w)
            coords[..., 1] = coords[..., 1] * (new_h / old_h)
            return coords

        masks = []
        scores = []
        nlses = []
        for i in range(x.shape[0]):
            nls = uncertainty[i]
            nlses.append(nls.cpu())
            points = apply_coords(query_coordinates[i]).cuda()
            points = (points[..., :2], points[..., -1])
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.predictor.model.prompt_encoder(
                boxes=None,
                points=points,
                masks=None,
            )

            (
                low_res_masks,
                iou_predictions,
            ) = self.predictor.model.mask_decoder(
                image_embeddings=self.predictor.features[i : i + 1],
                image_pe=self.predictor.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )
            mask = self.predictor.model.postprocess_masks(
                low_res_masks,
                self.predictor.input_size,
                self.predictor.original_size,
            )

            best_masks = iou_predictions.max(1)[1]
            mask = mask[torch.arange(mask.shape[0]), best_masks].unsqueeze(1)
            iou_predictions = iou_predictions[
                torch.arange(mask.shape[0]), best_masks
            ].unsqueeze(1)

            mask = mask > self.predictor.model.mask_threshold
            masks.append(mask.flatten(0, 1))

            score = []
            for i in range(mask.shape[0]):
                nls_score = nls[mask.squeeze(1)[i]].mean()
                if nls_score.isnan():
                    nls_score = -torch.ones_like(nls_score)
                score.append(nls_score)
            score = torch.stack(score)
            scores.append(score)
        return {"pred_logits": scores, "pred_masks": masks, "uncert": nlses}

    def test_step(self, batch: Any, batch_idx: int):
        # ood inference
        x, target = batch
        target, instance_target, _ = (
            target["sem_seg"],
            target["id"],
            target["panoptic_id"],
        )

        (
            query_coordinates,
            nls_uncertainty,
            road_prediction,
        ) = self.test_proposal_sampler(x=x, target=instance_target)
        x = F.interpolate(
            x,
            size=(512, 1024),
            mode="bilinear",
            align_corners=False,
        )
        ood_output = self.forward(
            x,
            query_coordinates,
            uncertainty=nls_uncertainty,
            instance_target=instance_target,
        )

        uncertainty = list()
        instance_prediction = []
        for i in range(len(ood_output["pred_logits"])):
            logits = ood_output["pred_logits"][i]
            masks = ood_output["pred_masks"][i]

            keep_by_nms = self.batched_mask_nms(
                masks, logits, iou_threshold=self.nms_thresh
            )
            keep_by_nms = torch.Tensor(keep_by_nms).long()

            if self.ignore_mask is not None:
                keep_by_ignore = list()
                for j in keep_by_nms.tolist():
                    intersection = torch.sum(masks[j] * self.ignore_mask)
                    if intersection < 100:
                        keep_by_ignore.append(j)
            else:
                keep_by_ignore = keep_by_nms

            query_coordinates[i] = query_coordinates[i][keep_by_ignore]
            uncertainty.append(
                torch.einsum(
                    "q,qhw->hw",
                    logits[keep_by_ignore].cpu(),
                    masks[keep_by_ignore].float().cpu(),
                )
            )
            instance_prediction.append(
                {
                    "scores": logits[keep_by_ignore].cpu().numpy()[..., None],
                    "masks": masks[keep_by_ignore].cpu().numpy().astype(np.uint8),
                }
            )

        uncertainty = torch.stack(uncertainty)
        if self.minimal_uncert_value_instead_zero:
            uncertainty[uncertainty == 0] = self.minimal_uncert_value_instead_zero
        nls_uncert = torch.stack(ood_output["uncert"])
        uncertainty = uncertainty + nls_uncert

        self.metrics["instance"].add(instance_prediction, instance_target)
        self.metrics["ood"].add(uncertainty.reshape(-1), target.reshape(-1))


        def map_instance_id2color(arr):
            # Get the unique values in the array
            unique_values = np.unique(arr)
            # Create a dictionary mapping the unique values to indices
            value_to_index = {value: index for index, value in enumerate(unique_values)}
            # Use np.vectorize to map the values to indices
            mapped_arr = np.vectorize(lambda x: value_to_index[x])(arr)
            return mapped_arr.astype(np.uint8)

        if not self.visualize:
            return

    def on_test_epoch_end(self):
        results = self.metrics["ood"].value()
        results.update(self.metrics["instance"].value())
        # results.update(self.metrics["panoptic"].value())
        self.log_dict(results)
        # panoptic_results = self.metrics["panoptic"].value()
        # self.log_dict(panoptic_results)

        if self.image_logger is None:
            return
        self.log_image_table(
            self.test_log_table,
            self.image_logger,
            mode="test",
            threshold=results["threshold"],
        )

    def log_image_table(self, image_table, logger, mode, threshold=None):
        for idx in range(len(image_table)):
            uncertainty = image_table[idx]["uncertainty"]
            uncertainty = (
                (uncertainty - uncertainty.min())
                / (uncertainty.max() - uncertainty.min() + 1e-8)
            ) * 255
            uncertainty_image = PILImage.fromarray(
                uncertainty.astype(np.uint8), mode="L"
            )
            image_table[idx]["uncertainty"] = wandb.Image(uncertainty_image)
            image_table[idx]["image"] = self.normalize_image(image_table[idx]["image"])
            for k, v in image_table[idx].items():
                if isinstance(v, np.ndarray):
                    image = PILImage.fromarray(v, mode="RGB")
                    image_table[idx][k] = wandb.Image(image)
        df = pd.DataFrame(image_table)
        df = wandb.Table(dataframe=df)
        logger.log({f"{mode}_predictions": df})
        return list()

    def configure_optimizers(self):
        params = [
            {
                "params": self.sam.mask_decoder.class_prediction_head.parameters(),
                "lr": self.optimizer.lr,
                "name": "class_prediction_head",
            },
        ]
        optimizer = hydra.utils.instantiate(
            self.optimizer,
            params=params,
        )
        if "steps_per_epoch" in self.scheduler.scheduler.keys():
            self.scheduler.scheduler.steps_per_epoch = len(
                self.trainer.datamodule.train_dataloader()
            )
        if "max_iters" not in self.scheduler.scheduler.keys():
            self.scheduler.scheduler.max_iters = (
                len(self.trainer.datamodule.train_dataloader())
                * self.trainer.max_epochs
            )
        lr_scheduler = hydra.utils.instantiate(
            self.scheduler.scheduler, optimizer=optimizer
        )
        scheduler_config = {"scheduler": lr_scheduler}
        scheduler_config.update(self.scheduler.pytorch_lightning_params)
        return [optimizer], [scheduler_config]

    def direct_ood_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("bqc,bqhw->bhw", mask_cls, mask_pred)
        return semseg

    @property
    def image_logger(self):
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                logger = logger.experiment
                return logger
        else:
            return None

    def normalize_image(self, image):
        mean_std = self.trainer.datamodule.mean_std
        image = (image.transpose(1, 2, 0) * mean_std[1] + mean_std[0]) * 255
        return image.astype(np.uint8)

    def get_ood_instance_prediction_image(self, image, prediction, mask_opacity=0.6):
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().cpu().numpy()
        image = self.normalize_image(image)
        return overlay_mask_on_image(image, prediction, mask_opacity=mask_opacity)

    def get_prediction_image(self, image, prediction, trainid2color, mask_opacity=0.6):
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().cpu().numpy()
        prediction_color = trainid2color[prediction]
        image = self.normalize_image(image)
        return overlay_mask_on_image(image, prediction_color, mask_opacity=mask_opacity)

    def direct_ood_instance_inference(self, mask_cls, mask_pred):
        scores = F.softmax(mask_cls, dim=-1)[..., :-1]
        bs = scores.shape[0]
        instances = list()
        for b in range(bs):
            indices = scores[b] > 0.5
            if sum(indices) == 0:
                index = torch.zeros_like(mask_pred[0, 0, ...])
                instances.append(index)
                continue
            m_p = mask_pred[b, indices.squeeze(), ...]
            scores_per_image = scores[b][indices]
            positive_masks = (m_p > 0).float()
            mask_scores_per_image = positive_masks
            scores_per_image = scores_per_image[..., None, None] * mask_scores_per_image
            scores_per_image = torch.cat(
                (torch.zeros_like(scores_per_image[:1, ...]), scores_per_image), dim=0
            )
            _, index = scores_per_image.max(0)
            instances.append(index)
        instances = torch.stack(instances)
        return instances

    def ood_instance_inference(self, mask_cls, mask_pred):
        scores = F.softmax(mask_cls, dim=-1)[..., :-1].detach().cpu().numpy()
        masks = mask_pred > 0
        masks = masks.detach().cpu().numpy().astype("uint8")
        instances = [
            {"scores": scores[i], "masks": masks[i]} for i in range(len(scores))
        ]
        return instances

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        return semseg

    def batched_mask_nms(self, masks, logits, iou_threshold):
        sorted_indices = torch.sort(logits, descending=True)[1]
        removed_mask_indices = list()
        num_classes = 2  # predicted / not predicted
        # going from the most confident mask compute iou with other masks
        for highest_score_index in sorted_indices:
            if highest_score_index in removed_mask_indices:
                # if mask already removed - don't check it
                continue
            target_mask = masks[highest_score_index].view(-1)
            # search only masks that were not removed
            remaining_masks = list(set(range(len(masks))) - set(removed_mask_indices))
            for mask_index in remaining_masks:
                if highest_score_index == mask_index:
                    # do not compare the mask with itself
                    continue
                x = masks[mask_index].view(-1) + num_classes * target_mask
                bincount_2d = torch.bincount(x, minlength=num_classes**2)

                true_positive = bincount_2d[3]
                false_positive = bincount_2d[1]
                false_negative = bincount_2d[2]
                # Just in case we get a division by 0, ignore/hide the error
                # no intersection
                if true_positive == 0:
                    continue
                # intersection of a source mask with the target mask
                iou = true_positive / (true_positive + false_positive + false_negative)
                if iou > iou_threshold:
                    removed_mask_indices.append(mask_index)
        resulting_masks = list(set(range(len(masks))) - set(removed_mask_indices))
        return resulting_masks

    @staticmethod
    def split_targets(targets, void_masks=False, remove_void=True):
        inlier_targets = []
        ood_targets = []

        for target in targets:
            labels = target["labels"]
            masks = target["masks"]

            if void_masks:
                labels[labels == 255] = 19

            inlier_mask = labels != 254
            if remove_void:
                inlier_mask = inlier_mask & (labels != 255)
            inlier_targets.append(
                {"labels": labels[inlier_mask], "masks": masks[inlier_mask]}
            )

            ood_mask = labels == 254
            ood_targets.append({"labels": labels[ood_mask], "masks": masks[ood_mask]})
        return inlier_targets, ood_targets
