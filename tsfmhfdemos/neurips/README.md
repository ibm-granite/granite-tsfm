# NeurIPS Expo 2023 Demo


## Installation
Running the demo requires first installing `tsfm_public` and then creating an appropriate set of models. At a later date we may release the pre-trained and finetuned models. To install the requirements use `pip` and then run the app using streamlit:


```bash
pip install ".[demos]"
cd tsfmhfdemos/neurips
streamlit run app.py
```

## Model structure

The models should be placed in the `neurips` folder in a folder named `models`. The folder is expected to have the folowing structure:
```
models
├── patchtsmixer
│   └── electricity
│       └── model
│           ├── pretrain
│           └── transfer
│               ├── ETTh1
│               │   ├── model
│               │   │   ├── fine_tuning
│               │   │   └── linear_probe
│               │   └── preprocessor
│               └── ETTh2
│                   ├── model
│                   │   ├── fine_tuning
│                   │   └── linear_probe
│                   └── preprocessor
└── patchtst
    └── electricity
        └── model
            ├── pretrain
            └── transfer
                ├── ETTh1
                │   ├── model
                │   │   ├── fine_tuning
                │   │   └── linear_probe
                │   └── preprocessor
                └── ETTh2
                    ├── model
                    │   ├── fine_tuning
                    │   └── linear_probe
                    └── preprocessor
```
The demo uses these pre-created models to perform inference and then plots and evaluates the forecasting results. The `pretrain`, `fine_tuning`, and `linear_probe` subfolders contain the serialized output from the appropriate Hugging Face model (using `Trainer.save_model()` or `TimeSeriesPreprocessor.save_pretrained()`).