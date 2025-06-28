# Copyright contributors to the TSFM project
#

import argparse
import os
import random
from collections import defaultdict
from functools import partial

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from utility.metrics import find_length_rank, get_metrics


# seeding
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def run_tsad_pipeline(
    data,
    label=None,
    num_input_channels=1,
    win_size=96,
    batch_size=256,
    smoothing_window=8,
    prediction_mode="time+fft+forecast",
    **kwargs,
):
    from utility.model import TSAD_Pipeline

    clf = TSAD_Pipeline(
        num_input_channels=num_input_channels,
        batch_size=batch_size,
        aggr_win_size=win_size,
        smoothing_window=smoothing_window,
        prediction_mode=prediction_mode,
    )
    clf.zero_shot(data, label)
    score = clf.decision_scores_
    return score.ravel()


def process_file(filename, args):
    all_results = defaultdict(list)
    print(f"Running: {filename}")
    print(args)
    try:
        df = pd.read_csv(os.path.join(args.data_direc, filename)).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df["Label"].astype(int).to_numpy()

        if data.ndim == 1:
            in_channels = 1
        else:
            in_channels = data.shape[1]

        slidingWindow = find_length_rank(data, rank=1)

        extra_kwargs = {}
        extra_kwargs.update(
            num_input_channels=in_channels,
            win_size=args.score_window,
            prediction_mode=args.mode,
            smoothing_window=args.smooth,
            batch_size=args.batch_size,
        )

        output = run_tsad_pipeline(data, **extra_kwargs)
        if isinstance(output, np.ndarray):
            output = MinMaxScaler(feature_range=(0, 1)).fit_transform(output.reshape(-1, 1)).ravel()

            evaluation_result = get_metrics(
                output,
                label,
                slidingWindow=slidingWindow,
                pred=output > (np.mean(output) + 2.5 * np.std(output)),
            )
            all_results["file"].append(filename)
            for k in evaluation_result.keys():
                all_results[k].append(evaluation_result[k])
        else:
            print(f"At {filename}: " + str(output))  # Ensure output is always string
        print(f"Ending: {filename}")
        return all_results
    except Exception:
        raise RuntimeError(f"Error: failed to process file {filename}!")


def parallel_process_files(all_files, args):
    # mpc.set_start_method('fork', force=True)
    partial_function = partial(process_file, args=args)
    # with mpc.Pool(8) as pool:
    #    results = pool.map(lfun, all_files)
    results = []
    for filename in all_files:
        try:
            result = partial_function(filename)
            results.append(result)
        except RuntimeError as e:
            print(e)
            pass

    final_all_results = defaultdict(list)

    for vus_pr in results:
        for k, v in vus_pr.items():
            final_all_results[k] += v

    return final_all_results


if __name__ == "__main__":
    ## ArgumentParser
    parser = argparse.ArgumentParser(description="Running TSB-AD")
    parser.add_argument(
        "--data_direc",
        type=str,
        default="Datasets/TSB-AD-U/",
        help="specify the directory where all the csv data-files are stored.",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default="Datasets/File_List/TSB-AD-U-Eva.csv",
        help="file containing list of valid csv files.",
    )
    parser.add_argument(
        "--out_file", type=str, default="TSB_results.csv", help="output file where the results will be stored."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default="#",
        help="optional file selector parameter, if specified runs experiment only on the files containing the string.",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="batch size used by the TSAD pipeline.")
    parser.add_argument(
        "--score_window", type=int, default=96, help="optional parameter to specify the scoring window size."
    )
    parser.add_argument(
        "--mode", type=str, default="forecast+fft+time", help="running mode for the TSPulse AD pipeline."
    )
    parser.add_argument("--smooth", type=int, default=8, help="score smoothing window specification.")
    args = parser.parse_args()
    print(args)

    file_filter = args.dataset if args.dataset != "#" else ".csv"
    if not os.path.isfile(args.eval_file):
        all_files = [filename for filename in os.listdir(args.data_direc) if file_filter in filename]
    else:
        all_files = []
        with open(args.eval_file, "r") as fp:
            all_files = [filename.strip() for filename in fp.readlines() if file_filter in filename.strip()]
    all_results = parallel_process_files(all_files, args)
    df = pd.DataFrame(all_results).set_index("file")
    df.to_csv(args.out_file, float_format="%.5f", header=True)
