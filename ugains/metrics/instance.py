import contextlib
import io
import json
import logging
import os
import tempfile
from collections import OrderedDict, namedtuple
from pathlib import Path

import numpy as np

import torch
from PIL import Image


class CityscapesEvaluator:
    def __init__(self, data_dir):
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self.data_dir = Path(data_dir)
        self.reset()

    def reset(self):
        self._working_dir = tempfile.TemporaryDirectory(prefix="cityscapes_eval_")
        self._temp_dir = self._working_dir.name
        # All workers will write to the same results directory
        # TODO this does not work in distributed training
        if self._temp_dir != self._working_dir.name:
            self._working_dir.cleanup()
        self._logger.info(
            "Writing cityscapes results to temporary directory {} ...".format(
                self._temp_dir
            )
        )


class CityscapesInstanceEvaluator(CityscapesEvaluator):
    """
    Evaluate instance segmentation results on cityscapes dataset using cityscapes API.

    Note:
        * It does not work in multi-machine distributed training.
        * It contains a synchronization, therefore has to be used on all ranks.
        * Only the main process runs evaluation.
    """

    def __init__(self, data_dir):
        super().__init__(data_dir)

    def add(self, prediction, targets):
        for prediction, target in zip(prediction, targets):
            basename = os.path.splitext(os.path.basename(target))[0]
            pred_txt = Path(self._temp_dir) / (basename + "_pred.txt")

            num_instances = len(prediction["scores"])
            with open(pred_txt, "w") as fout:
                for i in range(num_instances):
                    pred_score = prediction["scores"][i][0]
                    pred_class = 26
                    mask = prediction["masks"][i]
                    png_filename = os.path.join(
                        self._temp_dir, basename + f"_{i}_{pred_class}.png"
                    )
                    Image.fromarray(mask * 255).save(png_filename)
                    fout.write(
                        "{} {} {}\n".format(
                            os.path.basename(png_filename), pred_class, pred_score
                        )
                    )

    def value(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP" and "AP50".
        """
        import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as cityscapes_eval

        self._logger.info(f"Evaluating results under {self._temp_dir} ...")

        CsFile = namedtuple(
            "csFile", ["city", "sequenceNb", "frameNb", "type", "type2", "ext"]
        )

        def get_fs_file_info(parts):
            parts = parts.name
            parts = parts.split("_")
            parts = parts[:-1] + parts[-1].split(".")
            city, rest = parts[:-5], parts[-5:]
            city = ["_".join(city)]
            city.extend(rest)
            return CsFile(*city)

        _temp_dir = Path(self._temp_dir)
        cityscapes_eval.args.predictionPath = str(_temp_dir.absolute())
        cityscapes_eval.args.predictionWalk = None
        cityscapes_eval.args.JSONOutput = False
        cityscapes_eval.args.colorized = False
        cityscapes_eval.args.gtInstancesFile = _temp_dir / "gtInstances.json"
        # important change
        cityscapes_eval.args.minRegionSizes = np.array([10, 10, 10])
        cityscapes_eval.getCsFileInfo = get_fs_file_info

        print(self.data_dir)
        targets_base = self.data_dir / "gtCoarse"
        groundTruthImgList = sorted(list(targets_base.glob("*/*/*instanceIds.png")))

        assert len(
            groundTruthImgList
        ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
            cityscapes_eval.args.groundTruthSearch
        )
        predictionImgList = []
        for gt in groundTruthImgList:
            predictionImgList.append(
                cityscapes_eval.getPrediction(gt, cityscapes_eval.args)
            )
        results = cityscapes_eval.evaluateImgLists(
            predictionImgList, groundTruthImgList, cityscapes_eval.args
        )["averages"]

        ret = OrderedDict()
        ret = {
            "instance_AP": results["allAp"] * 100,
            "instance_AP50": results["allAp50%"] * 100,
        }
        self._working_dir.cleanup()
        return ret


class CityscapesPanopticEvaluator(CityscapesEvaluator):
    def __init__(self, data_dir, mode="test"):
        super().__init__(data_dir)
        self._predictions = []
        self.gt_folder = Path(data_dir) / mode
        if mode == "*":
            json_file_name = "lost_found_panoptic.json"
        else:
            json_file_name = f"lost_found_panoptic_{mode}.json"
        self.gt_json_file = Path(data_dir) / json_file_name

    @staticmethod
    def id2rgb(id_map):
        if isinstance(id_map, np.ndarray):
            id_map_copy = id_map.copy()
            rgb_shape = tuple(list(id_map.shape) + [3])
            rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
            for i in range(3):
                rgb_map[..., i] = id_map_copy % 256
                id_map_copy //= 256
            return rgb_map
        color = []
        for _ in range(3):
            color.append(id_map % 256)
            id_map //= 256
        return color

    def add(self, prediction, targets):
        for prediction, target in zip(prediction, targets):
            image_id = Path(target.replace("_gtCoarse_panoptic.png", "")).stem
            image_filename = Path(str(image_id) + ".png")
            dest_path = Path(self._temp_dir) / image_filename
            # for i, l in enumerate(prediction):
            #     prediction[prediction == l] = 26000 + i  # making it a car class
            segments_info = list()
            non_empty_regions = np.unique(prediction[prediction != 0])
            for pan_lab in non_empty_regions:
                if pan_lab > 1000:
                    category_id = pan_lab // 1000
                else:
                    category_id = pan_lab
                segments_info.append(
                    {
                        "id": int(pan_lab),
                        "category_id": int(category_id),
                    }
                )

            labels = self.id2rgb(prediction)
            Image.fromarray(labels).save(dest_path)

            self._predictions.append(
                {
                    "image_id": str(image_id),
                    "file_name": str(image_filename),
                    "segments_info": segments_info,
                }
            )

    def value(self):
        import cityscapesscripts.evaluation.evalPanopticSemanticLabeling as cityscapes_eval

        pred_folder = Path(self._temp_dir)
        pred_json_file = pred_folder / "predictions.json"
        resultsFile = pred_folder / "resultPanopticSemanticLabeling.json"

        with open(self.gt_json_file) as f:
            json_data = json.load(f)
        json_data["annotations"] = self._predictions
        with open(pred_json_file, "w") as f:
            f.write(json.dumps(json_data))

        with contextlib.redirect_stdout(io.StringIO()):
            results = cityscapes_eval.evaluatePanoptic(
                str(self.gt_json_file),
                str(self.gt_folder),
                str(pred_json_file),
                str(pred_folder),
                resultsFile,
            )
        print(results)
        ret = {
            "PQ": results["All"]["pq"] * 100,
            "RQ": results["All"]["rq"] * 100,
            "SQ": results["All"]["sq"] * 100,
        }
        self._working_dir.cleanup()
        return ret
