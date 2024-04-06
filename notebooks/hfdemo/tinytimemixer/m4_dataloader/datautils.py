# Standard
from types import SimpleNamespace

# First Party
from notebooks.hfdemo.tinytimemixer.m4_dataloader.data_factory import data_provider

seasonal_pattern = {
    "m4_hourly": "Hourly",
    "m4_daily": "Daily",
    "m4_weekly": "Weekly",
    "m4_monthly": "Monthly",
    "m4_quarterly": "Quarterly",
    "m4_yearly": "Yearly",
}


def get_m4_dataloaders(
    dataset_name,
    context_length,
    forecast_length,
    batch_size=32,
    num_workers=4,
    scale=True,
):
    args_ = {
        "data": "m4",
        "seasonal_patterns": seasonal_pattern[dataset_name],
        "features": "M",
        "embed": None,
        "percent": None,
        "batch_size": batch_size,
        "freq": None,
        "task_name": "short_term_forecast",
        "root_path": "datasets/m4",
        "data_path": None,
        "seq_len": context_length,
        "label_len": 0,
        "pred_len": forecast_length,
        "target": None,
        "num_workers": num_workers,
        "scale": scale,
    }
    args_ = SimpleNamespace(**args_)
    train_data, train_loader = data_provider(args_, "train")
    test_data, test_loader = data_provider(args_, "test")
    return train_data, train_loader, test_data, test_loader
