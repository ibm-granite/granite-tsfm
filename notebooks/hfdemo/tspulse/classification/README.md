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

Create a `seperate virtual environment` and pip install the `granite-tsfm` library using the following code snippet. Now pip install `specific versions` of the `torch` and `transformers` libraries as mentioned in the `../tspulse_repro_libs.txt`.

```bash
pip install git+https://github.com/ibm-granite/granite-tsfm.git@v0.2.28
pip install -r ../tspulse_repro_libs.txt
```
📌 **Note on Versioning for Reproducibility**
>
> To ensure reproducibility of the reported results, we have **fixed the versions** of `torch` and `transformers` libraries. Please use the specified versions in the `../tspulse_repro_libs.txt`, as different versions may lead to variations in numbers.

## Run classification benchmarking
Run the bash script `full_benchmarking_script.sh`. 
```bash
bash full_benchmarking_script.sh
```
It will run TSPulse classification benchmarking on all the datasets in the UEA classification archive mentioned in the bash script and save all the results in `tspulse_uea_classification_accuracies.csv`.