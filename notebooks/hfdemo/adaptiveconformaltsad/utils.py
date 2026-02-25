import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch

def create_rolling_forecast_contexts(
    past_values,
    context_length: int,
    prediction_length: int,
    max_context: int | None = None,  # cap for expanding; also validated for fixed
    mode: str = "fixed",  # "fixed" or "expanding"
    step: int = 1,  # stride between contexts
    return_type: str = "torch",  # "torch" or "numpy"
    dtype=torch.float32,
):
    """
    Create contexts and aligned ground-truth horizons.

    Args:
      past_values       : array-like (T,)
      context_length    : context length (fixed length if mode="fixed", minimum if mode="expanding")
      prediction_length : forecast horizon (H)
      max_context       : optional maximum context length (cap). If None, unbounded.
      mode              : "fixed" -> always length == context_length
                          "expanding" -> length grows with t, clipped by max_context,
                                         and emitted only when t >= context_length
      step              : stride between successive end indices
      return_type       : "torch" or "numpy" for outputs
      dtype             : torch dtype for tensors when return_type="torch"

    Returns:
      contexts : list of contexts (len N), each shape (L_i,) where
                 L_i == context_length if mode="fixed", else L_i ∈ [context_length, max_context]
      y_true   : (N, H) GT matrix, NaN where future values unavailable
      end_idx  : (N,) indices of the last obs included in each context
      n_valid  : (N,) number of valid GT steps per row
      ctx_len  : (N,) length of each emitted context (useful when mode="expanding")
    """
    pv = np.asarray(past_values)
    if pv.ndim != 1:
        raise ValueError("past_values must be 1D or 2D")

    T = int(pv.shape[0])
    H = int(prediction_length)
    if T < context_length:
        raise ValueError(f"Need at least context_length={context_length} points, got {T}")

    if max_context is not None:
        if max_context < context_length:
            raise ValueError(
                f"max_context ({max_context}) must be >= context_length ({context_length})."
            )
    assert mode in ["fixed", "expanding"], f"Invalid mode '{mode}'. Must be 'fixed' or 'expanding'."

    end_idx = np.arange(context_length - 1, T, step)
    N = len(end_idx)

    # allocate outputs
    if return_type == "torch":
        y_true = torch.full((N, H), float("nan"), dtype=dtype)
        n_valid = torch.zeros(N, dtype=torch.int64)
        ctx_len = torch.zeros(N, dtype=torch.int64)
    else:
        y_true = np.full((N, H), np.nan, dtype=float)
        n_valid = np.zeros(N, dtype=int)
        ctx_len = np.zeros(N, dtype=int)

    contexts = []
    for i, t_end in enumerate(end_idx):
        t = t_end + 1  # exclusive slice end

        if mode == "fixed":
            L = context_length
            if (max_context is not None) and (L > max_context):
                raise ValueError(
                    f"context_length ({context_length}) exceeds max_context ({max_context})."
                )
            start = t - L
        else:  # expanding
            # grow with t, but clip to max_context if provided
            L = t if max_context is None else min(t, max_context)
            # ensure we never drop below context_length (given t >= context_length, L>=context_length holds)
            L = max(L, context_length)
            start = t - L

        # context slice
        ctx_np = pv[start:t]
        if return_type == "torch":
            ctx = torch.tensor(ctx_np, dtype=dtype)
        else:
            ctx = ctx_np.astype(float, copy=False)
        contexts.append(ctx)

        # ground truth horizon (NaN-padded)
        nv = min(H, T - t)
        if nv > 0:
            if return_type == "torch":
                y_true[i, :nv] = torch.tensor(pv[t : t + nv], dtype=dtype)
            else:
                y_true[i, :nv] = pv[t : t + nv]
        n_valid[i] = nv
        ctx_len[i] = L

    if return_type == "torch":
        end_idx = torch.tensor(end_idx, dtype=torch.int64)

    output = {}
    output["past_values"] = contexts
    output["future_values"] = y_true
    output["end_idx"] = end_idx
    output["n_gt_valid"] = n_valid
    output["context_valid"] = ctx_len

    return output

def plot_anomaly_detection(data, label, p_values, label_pred, threshold=None, output_dir='output'):
    """
    Create a figure with two subplots for anomaly detection visualization.
    
    Parameters:
    -----------
    data : np.ndarray
        Time series data (N x features)
    label : np.ndarray
        True anomaly labels (1 for anomaly, 0 for normal)
    p_values : np.ndarray
        P-values for all time points
    label_pred : np.ndarray
        Predicted anomaly labels (1 for detected outlier, 0 for normal)
    threshold : float, optional
        Detection threshold (default: None, will not plot threshold line if not provided)
    output_dir : str
        Directory to save the output figure (default: 'output')
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Top plot: Time series with anomaly labels and predictions
    time_index = np.arange(len(data))
    ax1.plot(time_index, data[:, 0], 'b-', linewidth=1, label='Time Series')
    
    # Shade anomaly regions (where label == 1)
    anomaly_regions = label == 1
    ax1.fill_between(time_index, data[:, 0].min(), data[:, 0].max(),
                     where=anomaly_regions, alpha=0.3, color='red',
                     label='True Anomalies')
    
    # Mark detected outliers (where label_pred == 1)
    detected_outliers = label_pred == 1
    ax1.scatter(time_index[detected_outliers], data[detected_outliers, 0],
               color='red', marker='o', s=50, zorder=5,
               label='Detected Outliers', edgecolors='black', linewidths=0.5)
    
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('Time Series with Anomaly Detection', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: P-values in log scale with threshold
    ax2.semilogy(time_index, p_values, 'g-', linewidth=1, label='P-values')
    
    # Mark threshold with dotted line (if provided)
    if threshold is not None:
        ax2.axhline(y=threshold, color='orange', linestyle='--', linewidth=2,
                   label=f'Threshold = {threshold:.4f}')
    
    # Mark detected outliers (where label_pred == 1)
    ax2.scatter(time_index[detected_outliers], p_values[detected_outliers],
               color='red', marker='o', s=50, zorder=5,
               label='Detected Outliers', edgecolors='black', linewidths=0.5)
    
    ax2.set_xlabel('Time Index', fontsize=12)
    ax2.set_ylabel('P-values (log scale)', fontsize=12)
    ax2.set_title('P-values with Detection Threshold', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'anomaly_detection_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    plt.show()
    
    return output_path