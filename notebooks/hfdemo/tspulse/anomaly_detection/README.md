# Running TSAD Benchmarks

## Download Datasets
Download the TSB UAD Benchmark Datasets (please refer [TSB-AD Github](https://github.com/TheDatumOrg/TSB-AD), and the [evaluation file list](https://github.com/TheDatumOrg/TSB-AD/tree/main/Datasets) . 

## Recommended setup
The base environment should contain python >= 3.11. The dependent packages can be installed via pip utility using the `granite-tsfm/notebooks/hfdemo/tspulse/tspulse_repro_libs.txt` file provided.
```bash
$ pip install -r granite-tsfm/notebooks/hfdemo/tspulse/tspulse_repro_libs.txt
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
$ python run_experiment.py --data_direc <PATH TO CSVs> --eval_file <PATH TO EVAL FILES> [--dataset <DATASET NAME>] [--mode <ANOMALY DETECTION MODE>] [--out_file <Output filename>]
```

Using the `run_experiment.py` we produce the benchmarking scores, kept in `benchmarks` directory. 
After downloading the Dataset the benchmark univaraite results can be reproduced by running following commands.
```bash
$ cd <PATH TO GRANITE-TSFM CLONE DIRECTORY>/notebooks/hfdemo/tspulse/anomaly_detection
$ python run_experiment.py --data_direc "Datasets/TSB-AD-U" --eval_file "Datasets/File_List/TSB-AD-U-Tuning.csv"  --mode "time" --out_file "benchmarks/TSB-AD-U-Tuning-time.csv"
$ python run_experiment.py --data_direc "Datasets/TSB-AD-U" --eval_file "Datasets/File_List/TSB-AD-U-Tuning.csv"  --mode "fft" --out_file "benchmarks/TSB-AD-U-Tuning-fft.csv"
$ python run_experiment.py --data_direc "Datasets/TSB-AD-U" --eval_file "Datasets/File_List/TSB-AD-U-Tuning.csv"  --mode "forecast" --out_file "benchmarks/TSB-AD-U-Tuning-future.csv"
$ python run_experiment.py --data_direc "Datasets/TSB-AD-U" --eval_file "Datasets/File_List/TSB-AD-U-Tuning.csv"  --mode "time+fft+forecast" --out_file "benchmarks/TSB-AD-U-Tuning-ensemble.csv"
$ python run_experiment.py --data_direc "Datasets/TSB-AD-U" --eval_file "Datasets/File_List/TSB-AD-U-Eva.csv"  --mode "time" --out_file "benchmarks/TSB-AD-U-Eva-time.csv"
$ python run_experiment.py --data_direc "Datasets/TSB-AD-U" --eval_file "Datasets/File_List/TSB-AD-U-Eva.csv"  --mode "fft" --out_file "benchmarks/TSB-AD-U-Eva-fft.csv"
$ python run_experiment.py --data_direc "Datasets/TSB-AD-U" --eval_file "Datasets/File_List/TSB-AD-U-Eva.csv"  --mode "forecast" --out_file "benchmarks/TSB-AD-U-Eva-future.csv"
$ python run_experiment.py --data_direc "Datasets/TSB-AD-U" --eval_file "Datasets/File_List/TSB-AD-U-Eva.csv"  --mode "time+fft+forecast" --out_file "benchmarks/TSB-AD-U-Eva-ensemble.csv"
```
If the resource files are saved in a different location than suggested, amend the commands accordingly.

Following commands will produce multivariate scores on the benchmark dataset.
```bash
$ cd <PATH TO GRANITE-TSFM CLONE DIRECTORY>/notebooks/hfdemo/tspulse/anomaly_detection
$ python run_experiment.py --data_direc "Datasets/TSB-AD-M" --eval_file "Datasets/File_List/TSB-AD-M-Tuning.csv"  --mode "time" --out_file "benchmarks/TSB-AD-M-Tuning-time.csv"
$ python run_experiment.py --data_direc "Datasets/TSB-AD-M" --eval_file "Datasets/File_List/TSB-AD-M-Tuning.csv"  --mode "fft" --out_file "benchmarks/TSB-AD-M-Tuning-fft.csv"
$ python run_experiment.py --data_direc "Datasets/TSB-AD-M" --eval_file "Datasets/File_List/TSB-AD-M-Tuning.csv"  --mode "forecast" --out_file "benchmarks/TSB-AD-M-Tuning-future.csv"
$ python run_experiment.py --data_direc "Datasets/TSB-AD-M" --eval_file "Datasets/File_List/TSB-AD-M-Tuning.csv"  --mode "time+fft+forecast" --out_file "benchmarks/TSB-AD-M-Tuning-ensemble.csv"
$ python run_experiment.py --data_direc "Datasets/TSB-AD-M" --eval_file "Datasets/File_List/TSB-AD-M-Eva.csv"  --mode "time" --out_file "benchmarks/TSB-AD-M-Eva-time.csv"
$ python run_experiment.py --data_direc "Datasets/TSB-AD-M" --eval_file "Datasets/File_List/TSB-AD-M-Eva.csv"  --mode "fft" --out_file "benchmarks/TSB-AD-M-Eva-fft.csv"
$ python run_experiment.py --data_direc "Datasets/TSB-AD-M" --eval_file "Datasets/File_List/TSB-AD-M-Eva.csv"  --mode "forecast" --out_file "benchmarks/TSB-AD-M-Eva-future.csv"
$ python run_experiment.py --data_direc "Datasets/TSB-AD-M" --eval_file "Datasets/File_List/TSB-AD-M-Eva.csv"  --mode "time+fft+forecast" --out_file "benchmarks/TSB-AD-M-Eva-ensemble.csv"
```

To reproduce the triangulation results can be reproduced using `triangulation_scoring.py` scripts.
```bash
$ python triangulation_scoring.py --root_directory benchmarks  --prefix "TSB-AD-U-" --suffix ".csv" --metric "VUS-PR" --eval_prefix "Eva" --tuning_prefix "Tuning" 
============================================================
Triangulation Results On Tuning Data
============================================================
                  fft    future      time  ensemble      best
Exathlon     0.728570  0.619865  0.692210  0.674035       fft
IOPS         0.283325  0.126695  0.292725  0.290825      time
LTDB         0.711490  0.209670  0.693960  0.706100       fft
MGAB         0.004040  0.004930  0.003680  0.003810    future
MITDB        0.009820  0.002170  0.011340  0.010930      time
MSL          0.520935  0.369955  0.555170  0.559975  ensemble
NAB          0.481438  0.178700  0.473532  0.481226       fft
NEK          0.268900  0.190920  0.265410  0.274070  ensemble
OPPORTUNITY  0.035850  0.034865  0.035825  0.037405  ensemble
SED          0.034960  0.058750  0.031980  0.033950    future
SMAP         0.607415  0.392945  0.688475  0.657260      time
SMD          0.839190  0.400566  0.832444  0.835722       fft
SVDB         0.329860  0.036705  0.316580  0.087405       fft
Stock        0.560800  0.934765  0.561055  0.590750    future
TAO          0.432040  0.959530  0.432280  0.522360    future
TODS         0.680185  0.608395  0.755995  0.752790      time
UCR          0.034195  0.015442  0.046618  0.040315      time
WSD          0.473968  0.262830  0.481766  0.517612  ensemble
YAHOO        0.228504  0.598282  0.216366  0.577196    future
============================================================
Triangulated VUS-PR: 0.480
```

Following command will reproduce the multivariate triangulation results.
```bash
$ python triangulation_scoring.py --root_directory benchmarks  --prefix "TSB-AD-M-" --suffix ".csv" --metric "VUS-PR" --eval_prefix "Eva" --tuning_prefix "Tuning" 
============================================================
Triangulation Results On Tuning Data
============================================================
               future      time  ensemble       fft      best
CATSv2       0.091360  0.084670  0.083780  0.083440    future
Exathlon     0.813535  0.959335  0.955755  0.961295       fft
GHL          0.011295  0.011880  0.011890  0.011910       fft
LTDB         0.178040  0.225790  0.189680  0.220720      time
MITDB        0.006825  0.017020  0.024660  0.020805  ensemble
MSL          0.111630  0.695785  0.659850  0.694470      time
OPPORTUNITY  0.007050  0.006950  0.007270  0.007270  ensemble
SMAP         0.051575  0.260385  0.278190  0.269370  ensemble
SMD          0.327895  0.586610  0.596800  0.591000  ensemble
SVDB         0.052670  0.301043  0.217003  0.315867       fft
TAO          0.951335  0.113605  0.471315  0.113600    future
============================================================
Triangulated VUS-PR: 0.361
```
