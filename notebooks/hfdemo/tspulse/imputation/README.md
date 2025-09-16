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
Run the bash script `imputation_zeroshot.sh`.
```bash
bash imputation_zeroshot.sh
```
This will run the zeroshot imputation experiments for all the different datasets, mask type and mask ratios and will save the results in `tspulse_zeroshot_imputation_results.csv`

## Finetuning Experiments:
Run the bash script `imputation_finetune.sh`.
```bash
bash imputation_finetune.sh
```
This will run the finetuned imputation experiments for all the different datasets, mask type and mask ratios and will save the results in `tspulse_finetuned_imputation_results.csv`