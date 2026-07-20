# Running Classification Benchmarking
## Download datasets

Can Download a single folder containing all the UEA Multivariate Dataset from: http://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_ts.zip

Can download individual zip files for each datasets from: https://www.timeseriesclassification.com/dataset.php

Place the Datasets folders (having .ts files) in the `granite-tsfm/notebooks/hfdemo/tspulse/classification/datasets` folder as shown below.

<pre>
datasets
├── ArticularyWordRecognition
│   ├── ArticularyWordRecognition_TEST.ts
│   └── ArticularyWordRecognition_TRAIN.ts
├── BasicMotions
│   ├── BasicMotions_TEST.ts
│   └── BasicMotions_TRAIN.ts


</pre> 

## Prepare the environment 

Create a separate virtual environment and pip install the `granite-tsfm` library using the following code snippet.
```bash
$ pip install "torch==2.4.0" "transformers>=4.44.0,<4.51.0" "tensorboardX==2.6.2.2" "granite-tsfm[testing]==0.2.28" statsmodels
```
Note that you **should not** use these package versions in production environments since they have been updated due to reported CVEs.

📌 **Note on Versioning for Reproducibility**
>
> To ensure reproducibility of the reported results, we have **fixed the versions** of `torch` and `transformers` libraries. Please use the specified versions above, as different versions may lead to variations in numbers.

## Run classification benchmarking
Run the bash script `full_benchmarking_script.sh`. 
```bash
bash full_benchmarking_script.sh
```
It will run TSPulse classification benchmarking on all the datasets in the UEA classification archive mentioned in the bash script and save all the results in `tspulse_uea_classification_accuracies.csv`.