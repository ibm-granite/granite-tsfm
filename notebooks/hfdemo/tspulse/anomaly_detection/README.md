# Running TSAD Benchmarks

## Download Datasets
Download the TSB UAD Benchmark Datasets (please refer [TSB-AD Github](https://github.com/TheDatumOrg/TSB-AD), and the [evaluation file list](https://github.com/TheDatumOrg/TSB-AD/tree/main/Datasets) . 

## Recommended setup
The base environment should contain python >= 3.11. The dependent packages can be installed via pip utility using the `requirement.txt` file provided.
```bash
$ pip install -r requirement.txt
```

The script expects the datasets and resource files from [TSB-AD Github](https://github.com/TheDatumOrg/TSB-AD)) to be placed in specific directory architecture, as. specified below. 
<pre>
Datasets
 ├── TSB-AD-U
 │      ├── Univariates CSVs  
 ├── TSB-AD-M
 │      ├── Multi-variate CSVs
 ├── File_List
 │      ├── Resource file list 
</pre>

For any other configuration, must be explicitly specified during the script invocation

```bash
$ cd <PATH TO GRANITE-TSFM CLONE DIRECTORY>/notebooks/hfdemo/tspulse/anomaly_detection
$ python run_experiment.py --data_direc <PATH TO CSVs> --eval_file <PATH TO EVAL FILES> [--dataset <DATASET NAME>] [--mode <ANOMALY DETECTION MODE>]
```

Execution output with ensemble mode (specified as "__time__ + __fft__ + __forecast__") on the evaluation files for univariate and multivariate data are `TSB_U_Eval.csv`, and `TSB_M_Eval.csv` respectively.  
