# Copyright contributors to the TSFM project
#

import argparse
import os
import re
from typing import Optional

import numpy as np
import pandas as pd


def select_result_files(root_dir: str, prefix: str, suffix: Optional[str] = ".csv"):
    assert os.path.isdir(root_dir)
    selected_files, mode = [], []
    match_pattern = f"{prefix}-(.*){suffix}"
    for file in os.listdir(root_dir):
        if file.startswith(prefix) and file.endswith(suffix):
            if match := re.match(match_pattern, file):
                selected_files.append(file)
                mode.append(match[1])
    return selected_files, mode


def compute_average_performance(
    filename: str,
    metric: str = "VUS-PR",
    return_size: bool = False,
):
    assert os.path.isfile(filename)
    df = pd.read_csv(filename, index_col=0)
    assert metric in df
    index_list = df.index.tolist()
    index_group = {}
    for index in index_list:
        dataset_name = index.split("_")[1]
        if dataset_name not in index_group:
            index_group[dataset_name] = []
        index_group[dataset_name].append(index)
    scores = {}
    for grp in index_group:
        scores[grp] = df.loc[index_group[grp], metric].mean()
    if return_size:
        return scores, {k: len(v) for k, v in index_group.items()}
    else:
        return scores


def triangulation_performance(
    root_directory: str,
    prefix: str,
    suffix: str = ".csv",
    metric: str = "VUS-PR",
    tuning_file_prefix: str = "Tuning",
    eval_file_prefix: str = "Eva",
):
    tuning_prefix = f"{prefix}{tuning_file_prefix}"
    eval_prefix = f"{prefix}{eval_file_prefix}"

    selected_files, modes = select_result_files(root_directory, prefix=tuning_prefix, suffix=suffix)
    tuning_performance = {}
    for m, f in zip(modes, selected_files):
        tuning_performance[m] = compute_average_performance(os.path.join(root_directory, f), metric=metric)
    tuning_performance = pd.DataFrame(tuning_performance)
    cols = tuning_performance.columns.tolist()
    tuning_performance["best"] = [cols[c] for c in np.argmax(tuning_performance.values, axis=1)]

    selected_files, modes = select_result_files(root_directory, prefix=eval_prefix, suffix=suffix)
    eval_performance, grp_size = {}, {}
    for m, f in zip(modes, selected_files):
        eval_performance[m], grp_size = compute_average_performance(
            os.path.join(root_directory, f), metric=metric, return_size=True
        )
    eval_performance = pd.DataFrame(eval_performance)

    score, size = 0, 0
    for d_index in eval_performance.index:
        sel_mode = tuning_performance.loc[d_index, "best"] if d_index in tuning_performance.index else "time"
        sz = grp_size.get(d_index, 1)
        score += eval_performance.loc[d_index, sel_mode] * sz
        size += sz

    return {"tuning": tuning_performance, "evaluation": eval_performance, "metric": score / size}


if __name__ == "__main__":
    ## ArgumentParser
    root_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks")
    univariate_prefix = "TSB-AD-U-"
    metric = "VUS-PR"
    suffix = ".csv"

    parser = argparse.ArgumentParser(description="Running TSB-AD")
    parser.add_argument(
        "--root_directory",
        type=str,
        default=root_directory,
        help="specify the directory where all the score files are stored.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="TSB-AD-U-",
        help="file name prefix for Univariate file selection.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=suffix,
        help="file name prefix for Multivariate file selection.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=metric,
        help="AD metric to report, default VUS-PR.",
    )
    parser.add_argument(
        "--eval_prefix",
        type=str,
        default="Eva",
        help="File prefix for Eval file selection.",
    )
    parser.add_argument(
        "--tuning_prefix",
        type=str,
        default="Tuning",
        help="File prefix for Tuning file selection.",
    )
    args = parser.parse_args()

    result = triangulation_performance(
        root_directory=args.root_directory,
        prefix=args.prefix,
        suffix=args.suffix,
        metric=args.metric,
        tuning_file_prefix=args.tuning_prefix,
        eval_file_prefix=args.eval_prefix,
    )
    print("=" * 60)
    print("Triangulation Results On Tuning Data")
    print("=" * 60)
    print(result["tuning"].sort_index())
    print("=" * 60)
    print(f"Triangulated {metric}: {result['metric']:0.3f}\n\n")
