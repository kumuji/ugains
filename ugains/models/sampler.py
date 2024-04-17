import torch
import numpy as np
from PIL import Image as PILImage
import torch.nn.functional as F
import random
from pytorch_lightning import LightningModule
from contextlib import nullcontext
from omegaconf import OmegaConf
from ugains.models.mask2former.swin import SwinTransformer
from ugains.models.mask2former.pixel_decoder.msdeformattn import (
    MSDeformAttnPixelDecoder,
)
from ugains.models.mask2former.mask2former_transformer_decoder import (
    MultiScaleMaskedTransformerDecoder,
)
from scipy.ndimage import distance_transform_bf
from third_party.pointnet2.pointnet2_utils import furthest_point_sample
from sklearn.cluster import KMeans


def random_fill_no_feats(func):
    # random fill without supervision
    def fill(*args, **kwargs):
        ood_query_coords = func(*args, **kwargs)
        for t_index in range(len(ood_query_coords)):
            if ood_query_coords[t_index].shape[0] == 50:
                # query is full - no need to sample more
                continue
            # fill features when needed
            id_sample_idx = list(range(1024 * 2048))
            guaranteed_feats = ood_query_coords[t_index]
            sample_idx = random.sample(id_sample_idx, k=(50 - len(guaranteed_feats)))
            # fill locations
            unique_sample_idx = torch.zeros((1024, 2048))
            unique_sample_idx.view(-1)[sample_idx] = 1
            query_coords = unique_sample_idx.nonzero()
            query_coords = torch.flip(query_coords, [1])
            padding = torch.ones_like(query_coords[:, :1])
            query_coords = torch.cat((query_coords, padding), dim=1).unsqueeze(1)
            if torch.any(ood_query_coords[t_index] > 0):
                ood_query_coords[t_index] = torch.cat(
                    (ood_query_coords[t_index], query_coords)
                )
            else:
                ood_query_coords[t_index] = query_coords

        # fill empty indices
        return ood_query_coords

    return fill


class NLSSegmentation(LightningModule):
    def __init__(self):
        super().__init__()
        self.backbone = SwinTransformer(
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=12,
            ape=False,
            drop_path_rate=0.3,
            patch_norm=True,
        )
        input_shape = {
            "res2": {"channels": 128, "stride": 4},
            "res3": {"channels": 256, "stride": 8},
            "res4": {"channels": 512, "stride": 16},
            "res5": {"channels": 1024, "stride": 32},
        }
        input_shape = OmegaConf.create(input_shape)
        self.pixel_decoder = MSDeformAttnPixelDecoder(
            input_shape=input_shape,
            transformer_dropout=0.0,
            transformer_nheads=8,
            transformer_dim_feedforward=1024,
            transformer_enc_layers=6,
            conv_dim=256,
            mask_dim=256,
            norm="GN",
            # deformable transformer encoder args
            transformer_in_features=["res5"],
            common_stride=4,
        )

        self.sem_seg_head = MultiScaleMaskedTransformerDecoder(
            in_channels=256,
            num_classes=19,
            hidden_dim=256,
            num_queries=100,
            nheads=8,
            dim_feedforward=2048,
            dec_layers=1,
            pre_norm=False,
            mask_dim=256,
            enforce_input_project=False,
            dropout=0.0,
            num_feature_levels=3,
        )

    def prediction_step(self, batch, batch_idx=None):
        with torch.no_grad():
            x = batch
            output = self.backbone(x)
            multi_scale_features, mask_features = self.pixel_decoder(output)

            output = self.sem_seg_head(
                x=multi_scale_features,
                mask_features=mask_features,
                freeze_wrapper=nullcontext,
            )
            uncertainty = self.nls_ood_inference(
                output["pred_logits"],
                F.interpolate(
                    output["pred_masks"],
                    size=(1024, 2048),
                    mode="bilinear",
                    align_corners=False,
                ),
            )
            output = self.semantic_inference(
                output["pred_logits"],
                F.interpolate(
                    output["pred_masks"],
                    size=(1024, 2048),
                    mode="bilinear",
                    align_corners=False,
                ),
            )
        return uncertainty, output

    def nls_ood_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        uncertainty = -torch.sum(semseg, dim=1)
        return uncertainty

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        return semseg


class NLSSampler:
    def __init__(self, model_path, thresh, ignore_mask, method="fps", num_points=50):
        self.method = method
        self.num_points = num_points
        self.masked = True
        self.ignore_mask = None
        if ignore_mask:
            self.ignore_mask = torch.load(ignore_mask)
        self.uncertainty_thresh = thresh
        self.nls_model = NLSSegmentation()
        self.nls_model.eval()
        self.nls_model.cuda()
        self.nls_model.load_state_dict(
            torch.load(model_path)["state_dict"],
            strict=False,
        )

    def __call__(self, x, target):
        ood_query_coords = list()

        uncertainties = []
        road_prediction = []
        for t_index in range(len(target)):
            with torch.no_grad():
                uncert, inlier_output = self.nls_model.prediction_step(
                    x[t_index : t_index + 1]
                )

            road_prediction.append((inlier_output.max(1)[1][0] == 0).cpu().numpy())
            uncertainties.append(uncert)
            uncert = uncert[0]
            if self.ignore_mask is not None:
                uncert = torch.where(
                    ~self.ignore_mask.to(uncert.device),
                    uncert,
                    torch.full_like(uncert, -30),
                )
            sample_idx = torch.nonzero(uncert >= self.uncertainty_thresh)
            if sample_idx.sum() < 1:
                sample_idx = torch.nonzero(uncert >= -15)
            sample_idx = torch.flip(sample_idx, [1])
            padding = torch.ones_like(sample_idx[:, :1])
            sample_idx = torch.cat((sample_idx, padding), dim=1)
            if self.method == "fps":
                sample_idx = sample_idx.float().contiguous().cuda()
                fps_idx = furthest_point_sample(sample_idx[None, ...], self.num_points)[
                    0
                ].long()
                sample_idx = sample_idx[fps_idx].unsqueeze(1)
            elif self.method == "kmeans":
                predictor = KMeans(n_clusters=self.num_points).fit(sample_idx)
                sample_idx = (
                    torch.Tensor(predictor.cluster_centers_).long().unsqueeze(1)
                )
            elif self.method == "concon":
                nls_image = (uncert.cpu().numpy() >= self.uncertainty_thresh).astype(
                    np.uint8
                )
                num_labels, labels = cv2.connectedComponents(image=nls_image)
                sample_idx = list()
                for each_label in range(1, num_labels):
                    sample_idx.append(labels == each_label)
                sample_idx = np.stack(sample_idx)
            elif self.method == "random":
                sample_idx = sample_idx[
                    np.random.choice(
                        list(range(len(sample_idx))),
                        size=self.num_points,
                        replace=False,
                    )
                ].unsqueeze(1)

            ood_query_coords.append(sample_idx)
            # self.current_step += 1
        uncertainties = torch.cat(uncertainties, dim=0)
        return ood_query_coords, uncertainties, road_prediction


class BBoxProposalSampler:
    def __init__(self):
        super().__init__()

    def __call__(self, target):
        ood_query_coords = list()

        for t_index in range(len(target)):
            unique_instances_image = torch.Tensor(
                np.asarray(PILImage.open(target[t_index])).copy()
            )
            ood_unique_instances = torch.unique(unique_instances_image)
            ood_unique_instances = ood_unique_instances[ood_unique_instances > 1000]
            if len(ood_unique_instances) == 0:
                ood_query_coords.append(torch.full((1, 2, 3), -1))
                continue
            batch_query_coordinates = list()

            for unique_instance in ood_unique_instances:
                unique_sample_idx = (unique_instances_image == unique_instance).squeeze(
                    0
                )
                # Find the indices of the non-zero elements in the binary image
                nonzero_indices = torch.nonzero(unique_sample_idx)
                # Find the minimum and maximum values of the row and column indices
                min_row = nonzero_indices[:, 0].min()
                max_row = nonzero_indices[:, 0].max()
                min_col = nonzero_indices[:, 1].min()
                max_col = nonzero_indices[:, 1].max()
                # Create a bounding box in the format XYXY
                query_coords = torch.stack(
                    [min_col, min_row, max_col, max_row]
                ).reshape(2, 2)
                batch_query_coordinates.append(query_coords)

            # if for one frame different number of samples per object - fill
            max_size = max(c.shape[0] for c in batch_query_coordinates)
            coordinates = []
            for coordinate in batch_query_coordinates:
                # populating with only positive points
                pad = torch.ones_like(coordinate[..., :1])
                coordinate = torch.cat((coordinate, pad), dim=-1)
                pad_size = max_size - coordinate.shape[0]
                if pad_size == 0:
                    coordinates.append(coordinate)
                    continue
                coordinates.append(
                    torch.cat(
                        (
                            coordinate,
                            torch.full(
                                (pad_size, 3),
                                -1,
                                dtype=query_coords.dtype,
                                device=query_coords.device,
                            ),
                        )
                    )
                )
            batch_query_coordinates = torch.stack(coordinates)
            ood_query_coords.append(batch_query_coordinates)

        return ood_query_coords


class NoFeatSinglePointProposalSampler:
    def __init__(self, position="random", min_num_points=5, max_num_points=5):
        self.position = position
        self.min_num_points = min_num_points
        self.max_num_points = max_num_points

    @random_fill_no_feats
    def __call__(self, target):
        # position: random, first, center, average
        ood_query_coords = list()

        for t_index in range(len(target)):
            unique_instances_image = torch.Tensor(
                np.asarray(PILImage.open(target[t_index])).copy()
            )
            ood_unique_instances = torch.unique(unique_instances_image)
            ood_unique_instances = ood_unique_instances[ood_unique_instances > 1000]
            if len(ood_unique_instances) == 0:
                ood_query_coords.append(torch.full((1, 3, 3), -1))
                continue
            batch_query_coordinates = list()

            for unique_instance in ood_unique_instances:
                unique_sample_idx = (unique_instances_image == unique_instance).squeeze(
                    0
                )
                if self.position == "center":
                    distance_transform = distance_transform_bf(
                        unique_sample_idx.numpy()
                    )
                    # Find the maximum value in the distance transform
                    max_value = np.max(distance_transform)
                    # Find the coordinates where the maximum value appears
                    coordinates = np.argwhere(distance_transform == max_value)
                    # Calculate the mean of the coordinates to get the geometric center
                    query_coords = np.mean(coordinates, axis=0)[None, ...]
                    if self.max_num_points > 1:
                        num_points = min(
                            random.randint(self.min_num_points, self.max_num_points),
                            len(coordinates),
                        )
                        sample_idx = random.sample(range(len(coordinates)), num_points)
                        query_coords = coordinates[sample_idx]
                    query_coords = torch.from_numpy(query_coords).long()
                elif self.position == "random":
                    num_points = min(
                        random.randint(self.min_num_points, self.max_num_points),
                        unique_sample_idx.sum(),
                    )
                    sample_idx = random.sample(
                        range(unique_sample_idx.sum()), k=num_points
                    )
                    query_coords = torch.zeros_like(unique_sample_idx)
                    query_coords = unique_sample_idx.nonzero()[sample_idx]
                else:
                    raise NotImplementedError
                query_coords = torch.flip(query_coords, [1])
                batch_query_coordinates.append(query_coords)

            # if for one frame different number of samples per object - fill
            max_size = max(c.shape[0] for c in batch_query_coordinates)
            coordinates = []
            for coordinate in batch_query_coordinates:
                # populating with only positive points
                pad = torch.ones_like(coordinate[..., :1])
                coordinate = torch.cat((coordinate, pad), dim=-1)
                pad_size = max_size - coordinate.shape[0]
                if pad_size == 0:
                    coordinates.append(coordinate)
                    continue
                coordinates.append(
                    torch.cat(
                        (
                            coordinate,
                            torch.full(
                                (pad_size, 3),
                                -1,
                                dtype=query_coords.dtype,
                                device=query_coords.device,
                            ),
                        )
                    )
                )
            batch_query_coordinates = torch.stack(coordinates)
            ood_query_coords.append(batch_query_coordinates)
        return ood_query_coords
