# Copyright contributors to the TSFM project
#
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import create_rolling_forecast_contexts
from chronos import BaseChronosPipeline
import numpy as np
import torch

CHRONOS_MODELS = {
    "chronos-bolt-small": {
        "model_checkpoint": "amazon/chronos-bolt-small",
        "max_context_length": 2048,
    },
    "chronos-2":{
        "model_checkpoint": "amazon/chronos-2",
        "max_context_length": 2048,
    },
}

def chronos_max_context(pipeline):
    """Extract maximum context length from Chronos pipeline config."""
    cfg = getattr(getattr(pipeline, "model", pipeline), "config", None)
    if cfg is None:
        return None

    # Chronos puts it under chronos_config.context_length
    cc = getattr(cfg, "chronos_config", None)
    if isinstance(cc, dict):
        return cc.get("context_length")
    if cc is not None:
        return getattr(cc, "context_length", None)

    # Fallbacks seen in some models
    for k in ("context_length", "max_context_length", "max_position_embeddings"):
        v = getattr(cfg, k, None)
        if v is not None:
            return v
    return None

def chronos_forecaster(
    df,
    timestamp_column,
    target_columns,
    model_name='chronos-bolt-small',
    model_checkpoint=None,
    context_length=50,
    prediction_length=15,
    fixed_context=False,
    # device_map="cpu",
    max_context=None,
    batch_size=128,
):
    """
    Chronos forecaster compatible with granite_tsfm_forecaster interface.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with time series data
    timestamp_column : str
        Name of timestamp column
    target_columns : list of str
        Names of target columns to forecast
    model_name : str, default='chronos-bolt-small'
        Name of Chronos model (used to lookup checkpoint if model_checkpoint is None)
    model_checkpoint : str, optional
        Path or HuggingFace model ID for Chronos model
    context_length : int, default=50
        Context window length
    prediction_length : int, default=15
        Forecast horizon
    fixed_context : bool, default=False
        If True, use fixed context window; if False, use expanding window
    device_map : str, default="cpu"
        Device for inference ("cpu" or "cuda")
    max_context : int, optional
        Maximum context length (auto-detected from model if None)
    batch_size : int, default=128
        Batch size for inference
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'y_pred': np.ndarray, shape (n_samples, prediction_length, n_features)
        - 'y_true': np.ndarray, shape (n_samples, prediction_length, n_features)
    """
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    # Get model checkpoint
    if model_checkpoint is None:
        if model_name in CHRONOS_MODELS:
            model_checkpoint = CHRONOS_MODELS[model_name]["model_checkpoint"]
        else:
            raise ValueError(f"Unknown model_name: {model_name}. Provide model_checkpoint explicitly.")
    
    # Load Chronos pipeline
    pipeline = BaseChronosPipeline.from_pretrained(
        model_checkpoint,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )
    
    # Determine max context
    if max_context is None:
        max_context = chronos_max_context(pipeline)
        if max_context is None:
            max_context = 2048
    print(f"Using max_context: {max_context}")
    
    # Extract target values
    past_values = df[target_columns].values
    if len(past_values.shape) == 1:
        past_values = past_values[:, np.newaxis]
    
    N, F = past_values.shape
    
    # Process each feature
    y_pred_list = []
    y_true_list = []
    
    for ix_f in range(F):
        timeseries_univ = past_values[:, ix_f]
        
        # Create rolling contexts
        output = create_rolling_forecast_contexts(
            timeseries_univ,
            context_length=context_length,
            prediction_length=prediction_length,
            max_context=max_context,
            mode="fixed" if fixed_context else "expanding",
            return_type="torch",
        )
        
        contexts_ixf = output["past_values"]
        target_values_ixf = output["future_values"]
        
        # Batch inference
        m_all = []
        num_contexts = len(contexts_ixf)
        
        for i in range(0, num_contexts, batch_size):
            batch_ctx = contexts_ixf[i : i + batch_size]
            
            # Get median prediction (quantile 0.5)
            _, batch_m = pipeline.predict_quantiles(
                inputs=batch_ctx,
                prediction_length=prediction_length,
                quantile_levels=[0.5],
            )
            
            # Convert to numpy
            batch_m = (
                batch_m.detach().cpu().numpy()
                if hasattr(batch_m, "detach")
                else np.asarray(batch_m)
            )
            m_all.append(batch_m)
        
        # Concatenate batch results
        out_m_ixf = np.concatenate(m_all, axis=0) if m_all else np.array([])
        y_pred_list.append(out_m_ixf[..., np.newaxis])
        
        # Convert target values to numpy
        target_np = (
            target_values_ixf.detach().cpu().numpy()
            if hasattr(target_values_ixf, "detach")
            else np.asarray(target_values_ixf)
        )
        y_true_list.append(target_np[..., np.newaxis])
    
    # Concatenate features
    y_pred = np.concatenate(y_pred_list, axis=-1)
    y_true = np.concatenate(y_true_list, axis=-1)
    
    return {'y_pred': y_pred, 'y_true': y_true[:, :y_pred.shape[1], :]}

