# Copyright contributors to the TSFM project
#

import numpy as np


from TSB_AD.evaluation.basic_metrics import basic_metricor, generate_curve
from sklearn import metrics


def get_scores_tsb_ad_eval(scores, label, thresholds=None, slidingWindow=100, version='opt', thre=250):
    output = {}

    grader = basic_metricor()

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
    
    _, _, _, _, _, _,VUS_ROC, VUS_PR = generate_curve(label.astype(int), scores, slidingWindow, version, thre)

    output["threshold_independent_metrics"] = {
        "AUC_ROC": metrics.roc_auc_score(label, scores),
        "AUC_PR": metrics.average_precision_score(label, scores),
        "VUS_ROC": VUS_ROC,
        "VUS_PR": VUS_PR
    }
    ######## THRESHOLD DEPENDENT #########
    threshold_metrics = [
        "PA-F1",
        "Affiliation-F",
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
                metric_value = grader.metric_PointF1PA(label, scores, preds=pred)
            if th_metric == "Affiliation-F":
                metric_value = grader.metric_Affiliation(label, scores, preds=pred)

            if metric_value >= best_threshold_metrics[th_metric + "_point"][th_metric]:
                best_threshold_metrics[th_metric + "_point"][th_metric] = metric_value
                best_threshold_metrics[th_metric + "_point"]["fpr"] = fpr
                best_threshold_metrics[th_metric + "_point"]["score_threshold"] = (
                    1 - th_effective
                )
                best_threshold_metrics[th_metric + "_point"]["threshold"] = 1 - th #p-value compatible
    output["threshold_dependent_metrics"] = best_threshold_metrics
    return output