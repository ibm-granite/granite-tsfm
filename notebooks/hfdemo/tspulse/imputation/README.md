# Running Imputation Experiments
## Download datasets

ETT datasets (ETTh1, ETTh2, ETTm1 and ETTm2) can be automatically downloaded and will be used from the `dataset_path` provided in the notebooks. 
For weather and electricity download the zip folders namely weather.zip and electricity.zip from https://drive.google.com/drive/folders/1vE0ONyqPlym2JaaAoEe0XNDR8FS_d322.
Create a folder named 'datasets' in the `imputation` folder. 
Place the extracted weather and electricity folders in the 'datasets' folder. 

The datasets folder structure with the csv files should be like:
<pre> 
datasets
├── electricity
│   └── electricity.csv
└── weather
    └── weather.csv
</pre>

## Zero-Shot Experiments:
Run the notebook imputation_zeroshot.ipynb and provide the dataset name (dset), mask ratios(m_r) and mask type(m_t) values in the notebook.

## Finetuning Experiments:
Run the notebook imputation_finetune.ipynb and provide the DATASET, mask_ratio and mask_type values in the notebook 