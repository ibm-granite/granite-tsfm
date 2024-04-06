# Source: https://github.com/thuml/Time-Series-Library/tree/main

# Standard
from types import SimpleNamespace

# Third Party
from data.usecases.public_data.gpt4ts_datasets.data_loader import (
    Dataset_Custom,
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_M4,
    MSLSegLoader,
    PSMSegLoader,
    SMAPSegLoader,
    SMDSegLoader,
    SWATSegLoader,
    UEAloader,
)
from data.usecases.public_data.gpt4ts_datasets.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    "ETTh1": Dataset_ETT_hour,
    "ETTh2": Dataset_ETT_hour,
    "ETTm1": Dataset_ETT_minute,
    "ETTm2": Dataset_ETT_minute,
    "custom": Dataset_Custom,
    "m4": Dataset_M4,
    "PSM": PSMSegLoader,
    "MSL": MSLSegLoader,
    "SMAP": SMAPSegLoader,
    "SMD": SMDSegLoader,
    "SWAT": SWATSegLoader,
    "UEA": UEAloader,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != "timeF" else 1
    percent = args.percent

    if flag == "test":
        shuffle_flag = False
        drop_last = True
        if args.task_name == "anomaly_detection" or args.task_name == "classification":
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.task_name == "anomaly_detection":
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )
        return data_set, data_loader
    elif args.task_name == "classification":
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len),
        )
        return data_set, data_loader
    else:
        if args.data == "m4":
            drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns,
            scale=args.scale,
        )
        batch_size = args.batch_size
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )
        return data_set, data_loader


if __name__ == "__main__":
    # args_ = {
    #     "data": "m4",
    #     "seasonal_patterns": "Yearly",
    #     "features": "M",
    #     "embed": None,
    #     "percent": None,
    #     "batch_size": 64,
    #     "freq": None,
    #     "task_name": "short_term_forecast",
    #     "root_path": "/dccstor/dnn_forecasting/FM/data/m4",
    #     "data_path": None,
    #     "seq_len": 12,
    #     "label_len": 6,
    #     "pred_len": 6,
    #     "target": None,
    #     "num_workers": 1,
    # }
    args_ = {
        "data": "m4",
        "seasonal_patterns": "Daily",
        "features": "M",
        "embed": None,
        "percent": None,
        "batch_size": 64,
        "freq": None,
        "task_name": "short_term_forecast",
        "root_path": "/dccstor/dnn_forecasting/FM/data/m4",
        "data_path": None,
        "seq_len": 28,
        "label_len": 0,
        "pred_len": 14,
        "target": None,
        "num_workers": 1,
        "scale": True,
    }
    args_ = SimpleNamespace(**args_)
    train_data, train_loader = data_provider(args_, "train")
    val_data, val_loader = data_provider(args_, "val")
    test_data, test_loader = data_provider(args_, "test")

    breakpoint()
