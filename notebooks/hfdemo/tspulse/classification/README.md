# Running Classification Experiments
## Download datasets

Can Download a single folder containing all the UEA Multivariate Dataset from: http://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_ts.zip

Can download individual zip files for each datasets from: https://www.timeseriesclassification.com/dataset.php

Place the Datasets folders (having .ts files) in the `notebooks/hfdemo/tspulse/classification/datasets` folder as shown below.

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

Create a seperate conda environment having the `specific versions` of the required libraries as mentioned in the `granite-tsfm/notebooks/hfdemo/tspulse/requirements.txt`.

```bash
conda create --name tspulse_classification python=3.11.0
conda activate tspulse_classification

pip install git+https://github.com/ibm-granite/granite-tsfm.git@v0.2.27
pip install -r ../requirements.txt
```


## Run classification benchmarking
Run the python script `full_benchmarking_script.py`. It will run TSPulse classification benchmarking on datasets in the UEA classification archive and save all the results in `tspulse_uea_classification_accuracies.csv`.