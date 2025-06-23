# Running TSAD Benchmarks

## Download Datasets
Download the TSB UAD Benchmark Datasets (please refer [TSB-AD Github](https://github.com/TheDatumOrg/TSB-AD), and the [evaluation file list](https://github.com/TheDatumOrg/TSB-AD/tree/main/Datasets) . 

## Recommended setup
The base environment should contain python >= 3.11. The dependent packages can be installed via pip utility using the `requirement.txt` file provided.
<pre>
$ pip install -r requirement.txt
</pre> 

The script expects the datasets and resource files from [TSB-AD Github](https://github.com/TheDatumOrg/TSB-AD)) to be placed in specific directory architecture, as. specified below. 
<pre>
Datasets
 |
 |--TSB-AD-U
 |     |-univariates CSVs
 |
 |--TSB-AD-M
 |.    |-multi-variate CSVs
 |
 |--File_List
       |-resource file list 
</pre>

For any other configuration, must be explicitly specified during the script invocation

<pre>
$ cd <PATH TO GRANITE-TSFM CLONE DIRECTORY>/notebooks/hfdemo/tspulse/anomaly_detection
$ python run_experiment.py --data_direc <PATH TO CSVs> --eval_file <PATH TO EVAL FILES> [--dataset <DATASET NAME>] [--mode <ANOMALY DETECTION MODE>]
</pre>


