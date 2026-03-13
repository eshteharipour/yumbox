import math
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    dcg_score,
    f1_score,
    mean_squared_error,
    ndcg_score,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    top_k_accuracy_score,
)

from yumbox.mlflow import cleanup_plots


def report_scores(
    true, pred, scores, avg: Literal["macro", "micro", "weighted"] | None = None
):
    """Deprecated, kept for backwards compatibility."""
    accuracy = accuracy_score(true, pred)

    precision = precision_score(true, pred, average=avg, zero_division=np.nan)
    recall = recall_score(true, pred, average=avg, zero_division=np.nan)
    f1 = f1_score(true, pred, average=avg, zero_division=np.nan)
    # average_precision = average_precision_score(true, scores, average=avg)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        # "average_precision": average_precision,
    }


def classification_scores(
    true,
    pred,
    avg: Literal["micro", "macro", "samples", "weighted", "binary"] | None = "binary",
    is_binary: bool | None = None,
):
    assert 1 <= len(pred.shape) <= 2, pred
    if len(pred.shape) == 2:
        pred = pred[:, 0]

    # Check if it's binary classification (only 0s and 1s)
    if is_binary == None:
        is_binary = set(np.unique(true)) <= {0, 1} and set(np.unique(pred)) <= {0, 1}

    accuracy = accuracy_score(true, pred)
    precision = precision_score(true, pred, average=avg, zero_division=np.nan)
    recall = recall_score(true, pred, average=avg, zero_division=np.nan)
    f1 = f1_score(true, pred, average=avg, zero_division=np.nan)

    # Calculate FPR and FNR for binary classification
    fpr = fnr = np.nan
    if is_binary:
        try:
            # Get confusion matrix
            tn, fp, fn, tp = confusion_matrix(true, pred).ravel()

            # Calculate False Positive Rate (FPR) and False Negative Rate (FNR)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        except ValueError:
            # Handle case where confusion matrix can't be calculated (e.g., only one class present)
            fpr = fnr = np.nan

    # TODO: micro is duplicate of weighted on binary classification

    if is_binary:
        # For binary classification, return single accuracy without avg suffix
        return {
            "accuracy": accuracy,
            f"precision-{avg}": precision,
            f"recall-{avg}": recall,
            f"f1-{avg}": f1,
            "fpr": fpr,
            "fnr": fnr,
        }
    else:
        # For multi-class, include avg suffix for all metrics
        # Note: FPR and FNR are primarily meaningful for binary classification
        return {
            f"accuracy-{avg}": accuracy,
            f"precision-{avg}": precision,
            f"recall-{avg}": recall,
            f"f1-{avg}": f1,
        }


def extended_classification_scores(
    true: np.ndarray,
    pred: np.ndarray,
    topk: tuple | None = (1, 5),
    avgs: list[Literal["micro", "macro", "samples", "weighted", "binary"]] | None = [
        "binary",
        "micro",
        "macro",
        "weighted",
    ],
    is_binary: bool | None = None,
):
    """
    Extends sklearn_scores with top-k accuracies.

    Args:
        true: True labels (shape: [batch_size])
        pred: Predicted labels (shape: [batch_size] or [batch_size, n_classes])
        topk: Tuple of k values to compute top-k accuracies for

    Returns:
        Dictionary with extended metrics
    """

    # Calculate top-k accuracies
    topk_accs = []
    if topk:
        if len(pred.shape) == 2 and pred.shape[1] >= max(topk):
            topk_accs = np_topk_accuracy(true, pred, topk=topk)

    # Combine results
    extended_results = {
        **(
            {f"top-{k}_accuracies": acc for k, acc in zip(topk, topk_accs)}
            if topk
            else {}
        ),
    }

    for avg in avgs:
        extended_results.update(
            classification_scores(true, pred, avg=avg, is_binary=is_binary)
        )

    return extended_results


def torch_topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k.

    This function calculates the top-k accuracy metric for model predictions,
    measuring the percentage of samples where the correct label appears in the
    top k predictions.

    Args:
        output (torch.Tensor): Model output tensor of shape (batch_size, num_classes)
            containing the raw prediction scores (logits) for each class.
        target (torch.Tensor): Target tensor of shape (batch_size,) containing
            the ground truth class indices.
        topk (tuple, optional): Tuple of integers specifying which top-k accuracies
            to compute. Defaults to (1,) meaning only top-1 accuracy is computed.

    Returns:
        list: List of torch.Tensor objects, each containing the accuracy percentage
            for the corresponding k value in topk. Each tensor has shape (1,).

    Example:
        >>> output = torch.tensor([[0.5, 0.2, 0.3], [0.1, 0.6, 0.3]])
        >>> target = torch.tensor([0, 1])
        >>> torch_topk_accuracy(output, target, topk=(1, 2))
        [tensor([100.]), tensor([100.])]
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def np_topk_accuracy(true: np.ndarray, pred: np.ndarray, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    Assumes predected labels are sorted (best to worst).

    Args:
        true: Array of true labels (shape: [batch_size])
        pred: Array of predicted labels (shape: [batch_size, n_classes])
        topk: Tuple of k values to compute accuracies for

    Returns:
        List of accuracies for each k in topk (as percentages)
    """
    assert len(pred.shape) == 2
    assert pred.shape[1] >= max(topk)

    batch_size = true.shape[0]

    # Convert true labels to match pred shape for comparison
    true_expanded = np.expand_dims(true, axis=1)

    # Check if true label is in top k predictions
    res = []
    for k in topk:
        # Take first k predictions
        pred_k = pred[:, :k]
        # Check if true label is in predicted top k
        correct_k = np.any(pred_k == true_expanded, axis=1)
        # Calculate accuracy
        acc_k = np.sum(correct_k) / batch_size
        res.append(acc_k)

    return res


class Scorer:
    """Class for calculating various scoring metrics."""

    def __init__(
        self,
        sorted_labels: list[int],
        k: int = 0,
        total_positives: int = None,
        total_negatives: int = None,
    ):
        self.sorted_labels = sorted_labels
        self.total_positives = (
            total_positives  # Total number of true matches in the dataset
        )
        self.total_negatives = (
            total_negatives  # Total number of true non-matches in the dataset
        )
        self.set_k(k)

    def set_k(self, k: int):
        self.k = k if 0 < k <= len(self.sorted_labels) else len(self.sorted_labels)

    def precision(self) -> float:
        relevant_count = sum(1 for x in self.sorted_labels[: self.k] if x >= 1)
        return float(relevant_count) / self.k

    def average_precision(self) -> float:
        total_relevant = sum(1 for x in self.sorted_labels if x > 0)
        if total_relevant == 0:
            return 0.0

        ap_sum = 0.0
        relevant_count = 0

        for i in range(self.k):
            if self.sorted_labels[i] >= 1:
                relevant_count += 1
                ap_sum += float(relevant_count) / (i + 1.0)

        return ap_sum / total_relevant

    def recall(self) -> float:
        if self.total_positives is None:
            return -1.0

        true_positives = sum(1 for x in self.sorted_labels[: self.k] if x >= 1)
        return (
            true_positives / self.total_positives if self.total_positives > 0 else 0.0
        )

    def accuracy(self) -> float:
        correct_count = sum(1 for x in self.sorted_labels[: self.k] if x >= 1)
        return float(correct_count) / self.k

    def top_k_accuracy(self) -> float:
        return 1 if 1 in self.sorted_labels[: self.k] else 0

    def reciprocal_rank(self) -> float:
        for i, label in enumerate(self.sorted_labels):
            if label >= 1:
                return 1.0 / (i + 1)
        return 0.0

    def discounted_cumulative_gain(self) -> float:
        dcg_part = [
            (2 ** max(rel, 0) - 1) / math.log(index + 2, 2)
            for index, rel in enumerate(self.sorted_labels[: self.k])
        ]
        return sum(dcg_part)

    def normalized_discounted_cumulative_gain(self) -> float:
        dcg = self.discounted_cumulative_gain()
        ideal_labels = sorted(self.sorted_labels, reverse=True)
        ideal_dcg = sum(
            (2 ** max(rel, 0) - 1) / math.log(index + 2, 2)
            for index, rel in enumerate(ideal_labels[: self.k])
        )
        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def _mean_ir_scorer(
    queries: Iterable[Iterable[int]],
    metric: Literal[
        "TOPKACC", "ACC", "P", "AP", "R", "RR", "DCG", "NDCG", "FPR", "FNR"
    ],
    candidates_len: int,
    k=0,
    total_positives_per_query: list[int] = None,
    total_negatives_per_query: list[int] = None,
    sep: str = "@",
) -> dict[str, float]:
    """Returns a dict with the formatted metric name and its score based on the Scorer instance."""

    mapping = {
        "TOPKACC": {
            "f": "top_k_accuracy",
            "name": "mean-top-k-accuracy",  # Unused
            "formatted_name": "mean-top-{k}-accuracy",
        },
        "ACC": {
            "f": "accuracy",
            "name": "mean-accuracy",
            "formatted_name": "mean-accuracy{sep}{k}",
        },
        "P": {
            "f": "precision",
            "name": "mean-precision",
            "formatted_name": "mean-precision{sep}{k}",
        },
        "AP": {
            "f": "average_precision",
            "name": "MAP",
            "formatted_name": "MAP{sep}{k}",
        },
        "R": {
            "f": "recall",
            "name": "mean-recall",
            "formatted_name": "mean-recall{sep}{k}",
        },
        "RR": {
            "f": "reciprocal_rank",
            "name": "mean-reciprocal-rank",
            "formatted_name": "mean-reciprocal-rank",
        },
        "DCG": {
            "f": "discounted_cumulative_gain",
            "name": "mean-dcg",
            "formatted_name": "mean-dcg{sep}{k}",
        },
        "NDCG": {
            "f": "normalized_discounted_cumulative_gain",
            "name": "mean-ndcg",
            "formatted_name": "mean-ndcg{sep}{k}",
        },
        "FPR": {
            "f": "false_positive_rate",
            "name": "mean-fpr",
            "formatted_name": "mean-fpr{sep}{k}",
        },
        "FNR": {
            "f": "false_negative_rate",
            "name": "mean-fnr",
            "formatted_name": "mean-fnr{sep}{k}",
        },
    }

    if metric not in mapping:
        raise ValueError(f"Unknown metric name: {metric}")

    metric_func = mapping[metric]["f"]
    metric_name = mapping[metric]["name"]
    formatted_name = mapping[metric]["formatted_name"]

    # Create a sample scorer to get the formatted name
    sample_scorer = Scorer(
        queries[0],
        k,
        total_positives=(
            total_positives_per_query[0] if total_positives_per_query else None
        ),
        total_negatives=(
            total_negatives_per_query[0] if total_negatives_per_query else None
        ),
    )

    metric_name = (
        formatted_name.format(sep=sep, k=k)
        if sample_scorer.k != candidates_len
        else metric_name
    )

    scores = []
    for i in range(len(queries)):
        scorer = Scorer(
            queries[i],
            k,
            total_positives=(
                total_positives_per_query[i] if total_positives_per_query else None
            ),
            total_negatives=(
                total_negatives_per_query[i] if total_negatives_per_query else None
            ),
        )
        score_func = getattr(scorer, metric_func)
        score = score_func()
        if score is not None:  # for R and AR
            scores.append(score)

    mean_score = sum(scores) / len(scores) if len(scores) else 0.0
    return {metric_name: mean_score}


def mean_ir_scores(
    true: np.ndarray,
    pred: np.ndarray,
    candidates_len: int = 0,
    k: int | list[int] | range = 1,
    total_positives_per_query: list[int] = None,
    total_negatives_per_query: list[int] = None,
    metrics: list[
        Literal["TOPKACC", "ACC", "P", "AP", "R", "RR", "DCG", "NDCG", "FPR", "FNR"]
    ] = [
        "TOPKACC",
        "ACC",
        "P",
        "AP",
        "R",
        "RR",
        "DCG",
        "NDCG",
        "FPR",
        "FNR",
    ],
) -> dict[str, float]:
    """
    Compute all IR ranking metrics by comparing true and predicted labels.
    Labels are considered relevant (1) if they match, irrelevant (0) if they don't.

    Args:
        true: Numpy array of true labels as strings (1D, shape: (n_samples,)).
        pred: Numpy array of predicted labels as strings (1D or 2D, shape: (n_samples,) or (n_samples, top_k_labels)).
        candidates_len: Total number of candidates considered.
        k: Number of top results to consider (0 means full list for all metrics except Reciprocal Rank).
        total_positives_per_query: List of total number of true matches for each query (required for FNR and correct recall).
        total_negatives_per_query: List of total number of true non-matches for each query (required for FPR).
        metrics: List of metrics to compute, now including "FPR" and "FNR".

    Returns:
        Dictionary mapping metric names to their computed scores.
    """

    # TODO: Support negative query mode -> report Fall-out (Specificity), NPV (negative predictive value)
    # NOTE: the labels must be cluster or category id's
    # Convert labels to binary relevance scores
    if len(pred.shape) == 1:
        # 1D case: direct comparison
        queries = (true == pred).astype(int)
        queries = np.expand_dims(queries, axis=1).tolist()
    else:
        # 2D case: check if true label is in pred's top_k sorted labels for each sample
        true_expanded = np.expand_dims(true, axis=1)
        queries = (true_expanded == pred).astype(int).tolist()

    if isinstance(k, (int, float, str)):
        k = [int(k)]
    elif hasattr(k, "__iter__"):
        k = list(int(_k) for _k in k)

    sep = "-"
    k = list(sorted(k))

    all_scores = {}

    # Validate inputs for FPR/FNR
    # if "FPR" in metrics or "FNR" in metrics:
    #     if "FPR" in metrics and total_negatives_per_query is None:
    #         raise ValueError(
    #             "total_negatives_per_query must be provided when calculating FPR"
    #         )
    #     if "FNR" in metrics and total_positives_per_query is None:
    #         raise ValueError(
    #             "total_positives_per_query must be provided when calculating FNR"
    #         )

    kless_metrics = ["RR"]
    for m in kless_metrics:
        if m in metrics:
            all_scores.update(
                _mean_ir_scorer(
                    queries,
                    m,
                    candidates_len,
                    total_positives_per_query=total_positives_per_query,
                    total_negatives_per_query=total_negatives_per_query,
                    sep=sep,
                )
            )

    large_k_metrics = ["TOPKACC"]
    for at_k in k:
        if at_k < 2:
            continue
        for m in large_k_metrics:
            if m in metrics:
                all_scores.update(
                    _mean_ir_scorer(
                        queries,
                        m,
                        candidates_len,
                        at_k,
                        total_positives_per_query=total_positives_per_query,
                        total_negatives_per_query=total_negatives_per_query,
                        sep=sep,
                    )
                )

    k_metrics = ["ACC", "P", "AP", "R", "DCG", "NDCG", "FPR", "FNR"]
    for at_k in k:
        for m in k_metrics:
            # We can skip if m == "AP" and at_k == 1
            if m in metrics:
                all_scores.update(
                    _mean_ir_scorer(
                        queries,
                        m,
                        candidates_len,
                        at_k,
                        total_positives_per_query=total_positives_per_query,
                        total_negatives_per_query=total_negatives_per_query,
                        sep=sep,
                    )
                )

    return all_scores


def save_and_log_plot(fig, filename: str, outdir: str):
    """Helper function to save a plot and log it as an MLflow artifact."""
    # os.makedirs(outdir, exist_ok=True)
    # filepath = os.path.join(outdir, filename)
    # fig.savefig(filepath)
    # plt.close(fig)
    # mlflow.log_artifact(filepath)
    mlflow.log_figure(fig, filename)
    # plt.close(fig) # tkinter issue main thread not in main loop during training
    cleanup_plots()


def extra_scores_plots_binary(
    scores: np.ndarray, true: np.ndarray, outdir: str = "plots"
) -> dict[str, float]:
    if len(scores.shape) == 2:
        # scores = scores[:, 1]  # Positive class
        # In some datasets like breast cancer dataset, 0 is positive
        raise ValueError(
            "Received a 2D scores array with shape (n_samples, 2), "
            "which likely contains predicted probabilities for both classes. "
            "Please provide a 1D array with scores for the positive class only."
        )
    assert len(scores) == len(true), f"{len(scores)} != {len(true)}"
    assert np.unique(true).size == 2, "True labels must be binary"

    metrics = {}
    metrics["average_precision"] = average_precision_score(true, scores)
    # mlflow.log_metric("average_precision", metrics["average_precision"])

    precision, recall, _thresholds = precision_recall_curve(true, scores, pos_label=1)
    pr_auc = auc(recall, precision)
    metrics["pr_auc"] = pr_auc

    fpr, tpr, _ = roc_curve(true, scores, pos_label=1)
    roc_auc = roc_auc_score(true, scores)
    metrics["roc_auc"] = roc_auc

    # Plot 1: Precision-Recall Curve
    fig1 = plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR Curve (AUC = {pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="best")
    plt.grid(True)
    save_and_log_plot(fig1, "precision_recall_curve.png", outdir)

    # Plot 2: ROC Curve
    fig2 = plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="best")
    plt.grid(True)
    save_and_log_plot(fig2, "roc_curve.png", outdir)

    return metrics


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        try:
            return self.sum / self.count
        except ZeroDivisionError:
            return float("nan")
