# Copyright contributors to the TSFM project
#
from pandas.core.frame import DataFrame
import numpy as np

import time
import pandas as pd
import json
import os
import random
import torch
import argparse

from tsfm_public.toolkit.w1acas import (
    get_forecast_conformal_adaptive_online_score,
)
from forecasters.granite_tsfm_forecaster import granite_tsfm_forecaster
from standard_evaluation import get_scores_eval
from utils import plot_anomaly_detection
from forecasters.chronos_forecaster import chronos_forecaster
from forecasters.tirex_forecaster import tirex_forecaster

if __name__ == "__main__":
    
    # Argument parser
    parser = argparse.ArgumentParser(description='W1ACAS Anomaly Detection Pipeline')
    
    # Dataset and model configuration
    parser.add_argument('--dataset_file_path', type=str,
                        default='dataset/672_YAHOO_id_122_WebService_tr_500_1st_857.csv',
                        help='Path to the dataset CSV file')
    parser.add_argument('--model_name', type=str, default='chronos-bolt-small',
                        choices=['ttm', 'flowstate', 'chronos-bolt-small', 'tirex'],
                        help='Model name to use for forecasting')
    parser.add_argument('--context_length', type=int, default=90,
                        help='Context window length')
    parser.add_argument('--prediction_length', type=int, default=15,
                        help='Forecast horizon')
    
    # W1ACAS parameters
    parser.add_argument('--significance_level', type=float, default=0.01,
                        help='Target significance level (alpha) for anomaly detection')
    parser.add_argument('--aggregation_forecast_horizon', type=str, default='Cauchy',
                        choices=['median', 'mean', 'min', 'max', 'Fisher', 'HMC', 'Tippett', 'Cauchy'],
                        help='Method to aggregate p-values across forecast horizons')
    parser.add_argument('--aggregation_features', type=str, default='Cauchy',
                        choices=['median', 'mean', 'min', 'max', 'Fisher', 'HMC', 'Tippett', 'Cauchy'],
                        help='Method to aggregate p-values across features')
    parser.add_argument('--nonconformity_score', type=str, default='absolute_error',
                        help='Type of nonconformity score to compute forecast errors')
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of optimization epochs for adaptive weight learning')
    parser.add_argument('--n_batch_update', type=int, default=10,
                        help='Batch size for updating adaptive weights')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for adaptive weight optimization')
    parser.add_argument('--prior_past_weights_value', type=str, default='proximity',
                        help='Initialization strategy for past weights (0, "proximity", or numeric value)')
    parser.add_argument('--return_weights', action='store_true',
                        help='If set, returns calibration scores and weights along with p-values')
    parser.add_argument('--tsb_ad_evaluation', action='store_true',
                        help='If set, also computes TSB-AD evaluation metrics')
    
    args = parser.parse_args()

    #### SEEDS ####
    seed = 42
    # Python built-in random
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # Extract arguments
    dataset_file_path = args.dataset_file_path
    model_name = args.model_name
    context_length = args.context_length
    prediction_length = args.prediction_length
    significance_level = args.significance_level
    aggregation_forecast_horizon = args.aggregation_forecast_horizon
    aggregation_features = args.aggregation_features
    nonconformity_score = args.nonconformity_score
    n_epochs = args.n_epochs
    n_batch_update = args.n_batch_update
    lr = args.lr
    prior_past_weights_value = args.prior_past_weights_value
    return_weights = args.return_weights
    tsb_ad_evaluation = args.tsb_ad_evaluation
    
    # Conditionally import TSB-AD functions if needed
    if tsb_ad_evaluation:
        from tsb_ad_evaluation import get_scores_tsb_ad_eval
        from TSB_AD.utils.slidingWindows import find_length_rank
    

    # Dataset configuration
    dataset_name = os.path.basename(dataset_file_path).replace('.csv', '')
    
    df = pd.read_csv(dataset_file_path).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df["Label"].astype(int).to_numpy()
    train_index = dataset_name.split(".")[0].split("_")[-3]
    train_index = int(train_index)
    
    df_input = df.iloc[:, :-1].copy()
    target_columns = df.columns[:-1]
    timestamp_column = 'index'
    df_input.insert(0, timestamp_column, range(len(df_input)))

    # W1ACAS parameters
    forecast_steps = prediction_length
    align_forecast = True
    
    # Create output directory structure
    output_base_dir = f"output/{dataset_name}/{model_name}_{context_length}_{prediction_length}/W1ACAS_{nonconformity_score}_{aggregation_features}"
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"\nOutput directory: {output_base_dir}")
    
    # Generate forecasts based on model type
    if model_name in ['ttm', 'flowstate']:
        forecast_output = granite_tsfm_forecaster(
            df_input,
            timestamp_column,
            target_columns,
            model_name=model_name,
            model_checkpoint=None,
            context_length=context_length,
            prediction_length=prediction_length,
            fixed_context=False
        )
        
        print(forecast_output.keys())
        for _ in forecast_output.keys():
            print(_, forecast_output[_].shape)
        
        # Save forecast output
        # forecast_output_path = os.path.join(output_base_dir, "forecast_output.npz")
        # np.savez(forecast_output_path, **forecast_output)
        # print(f"Forecast output saved to: {forecast_output_path}")
    elif 'chronos' in model_name:
        forecast_output = chronos_forecaster(
            df_input,
            timestamp_column,
            target_columns,
            model_name=model_name,
            model_checkpoint=None,
            context_length=context_length,
            prediction_length=prediction_length,
            fixed_context=False
        )
    elif model_name == 'tirex':
        forecast_output = tirex_forecaster(
            df_input,
            timestamp_column,
            target_columns,
            model_name=model_name,
            model_checkpoint=None,
            context_length=context_length,
            prediction_length=prediction_length,
            fixed_context=False
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}. Must be 'ttm' or 'flowstate'.")

    # Check if last row of y_true is all NaNs (no ground truth for future predictions)
    y_true = forecast_output['y_true']
    last_row_all_nan = np.all(np.isnan(y_true[-1]))
    
    if last_row_all_nan:
        print(f"Last row of y_true is all NaN - will be excluded from scoring")
        # Remove last row from forecast output before scoring
        forecast_output_filtered = {
            'y_pred': forecast_output['y_pred'][:-1],
            'y_true': forecast_output['y_true'][:-1]
        }
    else:
        forecast_output_filtered = forecast_output

    # Start timing the entire inference process
    inference_start_time = time.time()
    p_values = get_forecast_conformal_adaptive_online_score(
        forecast_output_filtered,
        significance_level=significance_level,
        aggregation_forecast_horizon=aggregation_forecast_horizon,
        nonconformity_score=nonconformity_score,
        forecast_steps=forecast_steps,
        aggregation_features=aggregation_features,
        n_epochs=n_epochs,
        n_batch_update=n_batch_update,
        lr=lr,
        prior_past_weights_value=prior_past_weights_value,
        return_weights=return_weights,
        align_forecast=align_forecast,
    )
    # End timing the entire inference process
    inference_time = time.time() - inference_start_time
    
    # Create p_values_all array (pad with p_value 1s for initial context)
    p_values_all = np.ones(len(df))
    p_values_all[-len(p_values):] = p_values
    p_values_all[:train_index] = 1
    p_values_test = p_values_all[train_index:]
    label_test = label[train_index:]
    # p_values_all[0:train_index] = 1

    print(f"P-values test shape: {p_values_test.shape}, label_test shape: {label.shape}")

    
    
    # Compute standard evaluation metrics
    evaluation = get_scores_eval(1 - p_values_test, label_test)
    print("\nStandard Evaluation metrics:")
    print(evaluation)
    
    # Get threshold and predictions
    threshold = evaluation['threshold_dependent_metrics']['PA-F1_point']['threshold']
    label_pred_test = p_values_test < threshold
    
    # Compute TSB-AD evaluation metrics if requested
    if tsb_ad_evaluation:
        slidingWindow = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
        evaluation_tsb_ad = get_scores_tsb_ad_eval(1 - p_values_test, label_test, slidingWindow=slidingWindow)
        print("\nTSB-AD Evaluation metrics:")
        print(evaluation_tsb_ad)

        # Save TSB-AD evaluation as JSON if computed
        evaluation_tsb_ad_path = os.path.join(output_base_dir, "evaluation_tsb_ad.json")
        with open(evaluation_tsb_ad_path, 'w') as f:
            json.dump(evaluation_tsb_ad, f, indent=2)
        print(f"TSB-AD Evaluation saved to: {evaluation_tsb_ad_path}")
    
    
    # Save p_values_all as CSV
    p_values_df = pd.DataFrame({
        'p_values_test': p_values_test,
        'gt_labels_test': label_test,
        'pred_labels_test': label_pred_test.astype(int),
        'inference_time': inference_time
    })
    p_values_path = os.path.join(output_base_dir, "p_values.csv")
    p_values_df.to_csv(p_values_path, index=False)
    print(f"P-values saved to: {p_values_path}")
    
    # Save standard evaluation as JSON
    evaluation_path = os.path.join(output_base_dir, "evaluation.json")
    with open(evaluation_path, 'w') as f:
        json.dump(evaluation, f, indent=2)
    print(f"Evaluation saved to: {evaluation_path}")

    # Generate and save visualization
    print("\nGenerating visualization...")
    plot_anomaly_detection(data[train_index:], label_test, p_values_test, label_pred_test, threshold=threshold, output_dir=output_base_dir)
    
    print(f"\n✓ All outputs saved to: {output_base_dir}")
