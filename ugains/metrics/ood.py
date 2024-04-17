import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_curve, auc

# import seaborn as sns


class OODMetrics:
    def __init__(self, ignore_label):
        self.ignore_label = ignore_label
        self.reset()

    @staticmethod
    def calculate_auroc(predictions, targets):
        fpr, tpr, thresholds = roc_curve(y_true=targets, y_score=predictions)
        roc_auc = auc(fpr, tpr)
        fpr_best = 0
        threshold = 0
        for i, j, threshold in zip(tpr, fpr, thresholds):
            if i > 0.95:
                fpr_best = j
                break
        return roc_auc, fpr_best, threshold

    def value(self):
        targets = np.concatenate(self.targets, axis=0)
        predictions = np.concatenate(self.predictions, axis=0)
        AP = average_precision_score(y_true=targets, y_score=predictions)
        roc_auc, fpr, threshold = self.calculate_auroc(
            predictions=predictions, targets=targets
        )
        return {
            "AP": AP * 100,
            "FPR95": fpr * 100,
            "AUROC": roc_auc * 100,
            "threshold": threshold,
        }

    def reset(self):
        self.predictions = []
        self.targets = []

    def add(self, predicted, target):
        if torch.is_tensor(predicted):
            predicted = predicted.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()

        ind = ~np.isin(target, self.ignore_label)
        predicted, target = predicted[ind], target[ind]

        assert (
            predicted.shape[0] == target.shape[0]
        ), "number of targets and predicted outputs do not match"

        self.targets.append(target)
        self.predictions.append(predicted)
