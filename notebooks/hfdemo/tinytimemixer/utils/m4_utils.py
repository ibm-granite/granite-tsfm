# Standard
import math
import os

# Third Party
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# First Party
from notebooks.hfdemo.tinytimemixer.m4_dataloader.m4_summary import M4Summary


def m4_visual(ax, true, preds, naive, border, index=None):
    """
    Results visualization
    """
    ax.plot(preds, label="Prediction", linestyle="--", color="orange", linewidth=2)
    ax.plot(naive, label="Naive2", linestyle=":", color="k", linewidth=2)
    ax.plot(true, label="GroundTruth", linestyle="-", color="blue", linewidth=2)
    ax.axvline(x=border, color="r", linestyle="-")
    ax.set_title(f"Example {index}")
    ax.legend(loc="center left")


# Pre-train
def m4_finetune(
    model,
    dset_train,
    lr=0.001,
    save_path="/tmp",
    num_epochs=20,
    batch_size=64,
    num_workers=4,
):
    trainer_args = TrainingArguments(
        output_dir=os.path.join(save_path, "checkpoint"),
        overwrite_output_dir=True,
        learning_rate=lr,
        num_train_epochs=num_epochs,
        do_eval=False,
        # evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        # per_device_eval_batch_size=batch_size,
        dataloader_num_workers=num_workers,
        ddp_find_unused_parameters=False,
        report_to="tensorboard",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        logging_dir=os.path.join(
            save_path, "logs"
        ),  # Make sure to specify a logging directory
        load_best_model_at_end=False,  # Load the best model when training ends
        # metric_for_best_model=metric_for_best_model,  # Metric to monitor for early stopping
        greater_is_better=False,  # For loss
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = OneCycleLR(
        optimizer,
        lr,
        epochs=num_epochs,
        steps_per_epoch=math.ceil(len(dset_train) / batch_size),
    )

    def collate_fn(examples):
        past_values = torch.stack([torch.Tensor(example[0]) for example in examples])
        future_values = torch.stack([torch.Tensor(example[1]) for example in examples])
        # print(past_values.shape, future_values.shape)
        return {"past_values": past_values, "future_values": future_values}

    # Set trainer
    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=dset_train,
        optimizers=(optimizer, scheduler),
        data_collator=collate_fn,
    )

    # Train
    trainer.train()

    # Save the pretrained model
    trainer.save_model(os.path.join(save_path, "ttm_pretrained"))


# Taken from: https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All/blob/main/Short-term_Forecasting/exp/exp_short_term_forecasting.py
def m4_test(
    model,
    train_loader,
    test_loader,
    save_path,
    model_prefix,
    device,
    forecast_length=48,
    seasonal_patterns="Hourly",
):
    x, _ = train_loader.dataset.last_insample_window()
    y = test_loader.dataset.timeseries
    x = torch.tensor(x, dtype=torch.float32).to(device)
    x = x.unsqueeze(-1)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model.eval()
    with torch.no_grad():
        # label_len = 0 for TTM
        label_len = 0
        B, _, C = x.shape
        dec_inp = torch.zeros((B, forecast_length, C)).float().to(device)
        dec_inp = torch.cat([x[:, -label_len:, :], dec_inp], dim=1).float()
        # encoder - decoder
        outputs = torch.zeros((B, forecast_length, C)).float().to(device)
        id_list = np.arange(0, B, 1024)  # batch size = 1024
        id_list = np.append(id_list, B)
        for i in range(len(id_list) - 1):
            # outputs[id_list[i] : id_list[i + 1], :, :] = self.model(
            #     x[id_list[i] : id_list[i + 1]], None, dec_inp[id_list[i] : id_list[i + 1]], None
            # )
            ttm_out = model(past_values=x[id_list[i] : id_list[i + 1]])
            outputs[id_list[i] : id_list[i + 1], :, :] = ttm_out.prediction_outputs

            if id_list[i] % 1000 == 0:
                print(id_list[i])

        # Rescale
        outputs = test_loader.dataset.inverse_transform(outputs)

        # f_dim = -1 if self.features == "MS" else 0
        f_dim = 0
        outputs = outputs[:, -forecast_length:, f_dim:]
        outputs = outputs.detach().cpu().numpy()

        # Rescale trues
        y = test_loader.dataset.inverse_transform(torch.Tensor(y))
        x = test_loader.dataset.inverse_transform(x)

        preds = outputs
        trues = y
        x = x.detach().cpu().numpy()

    print("test shape:", preds.shape)

    # result save
    folder_path = "./m4_results/" + "ttm" + "/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    forecasts_df = pd.DataFrame(
        preds[:, :, 0], columns=[f"V{i + 1}" for i in range(forecast_length)]
    )
    forecasts_df.index = test_loader.dataset.ids[: preds.shape[0]]
    forecasts_df.index.name = "id"
    forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
    forecasts_df.to_csv(folder_path + seasonal_patterns + "_forecast.csv")

    file_path = "./m4_results/" + "ttm" + "/"

    m4_summary = M4Summary(
        file_path,
        "datasets/m4",
    )
    # m4_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
    # smape_results, owa_results, mape, mase = m4_summary.evaluate()
    results = m4_summary.evaluate_single(seasonal_patterns)
    results.to_csv(f"{save_path}/results_{model_prefix}.csv")

    # Set a more beautiful style
    plt.style.use("seaborn-v0_8-whitegrid")
    # Adjust figure size and subplot spacing
    num_plots = 10
    fig, axs = plt.subplots(num_plots, 1, figsize=(10, 15))
    random_indices = np.random.choice(preds.shape[0], size=num_plots, replace=False)
    for idx, i in enumerate(random_indices):
        gt = np.concatenate((x[i, : forecast_length * 2, 0], trues[i]), axis=0)
        prd = np.concatenate((x[i, : forecast_length * 2, 0], preds[i, :, 0]), axis=0)
        naive = np.concatenate(
            (x[i, : forecast_length * 2, 0], m4_summary.naive2_forecasts[i, :]),
            axis=0,
        )
        m4_visual(
            ax=axs[idx],
            true=gt,
            preds=prd,
            naive=naive,
            border=forecast_length * 2,
            index=i,
        )
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"m4_{seasonal_patterns}_{model_prefix}.pdf"))

    # if (
    #     "Weekly_forecast.csv" in os.listdir(file_path)
    #     and "Monthly_forecast.csv" in os.listdir(file_path)
    #     and "Yearly_forecast.csv" in os.listdir(file_path)
    #     and "Daily_forecast.csv" in os.listdir(file_path)
    #     and "Hourly_forecast.csv" in os.listdir(file_path)
    #     and "Quarterly_forecast.csv" in os.listdir(file_path)
    # ):
    #     m4_summary = M4Summary(
    #         file_path,
    #         "datasets/m4",
    #     )
    #     # m4_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
    #     smape_results, owa_results, mape, mase = m4_summary.evaluate()
    #     print("*" * 20, "Final combined metrics across all of M4", "*" * 20)
    #     print("smape:", smape_results)
    #     print("mape:", mape)
    #     print("mase:", mase)
    #     print("owa:", owa_results)
    # else:
    #     print("After all 6 tasks are finished, you can calculate the averaged index")

    return
