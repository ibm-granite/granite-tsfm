
# Copyright contributors to the TSFM project
#
"""
Time Series Anomaly Detection Metrics

This module contains evaluation metrics for time series anomaly detection.

The `adjust_predicts` function is adapted from the TSB-AD repository:
https://github.com/TheDatumOrg/TSB-AD

Reference:
    Qinghua Liu and John Paparrizos.
    "The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark"
    NeurIPS 2024.
"""

from sklearn import metrics
import numpy as np
import math
import copy


def adjust_predicts(score, label, threshold=None, pred=None, calc_latency=False
):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    
    Adapted from TSB-AD: https://github.com/TheDatumOrg/TSB-AD

    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is higher than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):

    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = copy.deepcopy(pred)
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def metric_PointF1PA(score, label, preds):
    adjust_preds = adjust_predicts(score, label, pred=preds)
    PointF1PA1 = metrics.f1_score(label, adjust_preds)

    return PointF1PA1


def get_scores_eval(scores, label, thresholds=None):
    output = {}
    if thresholds is None:
        alpha_p_values = np.concatenate(
            [
                np.linspace(0.001, 0.01, 21),  # Fine-grained low α values
                np.linspace(0.02, 0.1, 21),  # Medium range
                np.linspace(0.2, 1, 21),  # High α for broader picture
            ]
        )
        thresholds = np.sort(
            np.unique(np.sort(np.concatenate([alpha_p_values, 1 - alpha_p_values])))
        )

    output["threshold_independent_metrics"] = {
        "AUC_ROC": metrics.roc_auc_score(label, scores),
        "AUC_PR": metrics.average_precision_score(label, scores),
    }

    ######## THRESHOLD DEPENDENT #########
    threshold_metrics = [
        "PA-F1",
    ]
    best_threshold_metrics = {}
    for th_metric in threshold_metrics:
        best_threshold_metrics[th_metric + "_point"] = {}
        best_threshold_metrics[th_metric + "_point"][th_metric] = 0

    for th in thresholds:
        if th >= 1:
            continue
        pred = np.array(scores >= th).astype("int")
        if np.sum(pred) <= 0:
            continue
        th_effective = np.min(scores[pred == 1])
        fpr = np.mean(pred[label == 0])
        for th_metric in threshold_metrics:
            metric_value = 0
            if th_metric == "PA-F1":
                metric_value = metric_PointF1PA(scores, label, preds=pred
                )
            if metric_value >= best_threshold_metrics[th_metric + "_point"][th_metric]:
                best_threshold_metrics[th_metric + "_point"][th_metric] = metric_value
                best_threshold_metrics[th_metric + "_point"]["fpr"] = fpr
                best_threshold_metrics[th_metric + "_point"]["score_threshold"] = (
                    1 - th_effective
                )
                best_threshold_metrics[th_metric + "_point"]["threshold"] = 1 - th #p-value compatible
    output["threshold_dependent_metrics"] = best_threshold_metrics
    return output