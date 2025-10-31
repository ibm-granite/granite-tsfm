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

The `run_experiment.py` script also support model finetuning. following commands will produce univariate finetuning scores on the benchmark dataset.
```bash
$ cd <PATH TO GRANITE-TSFM CLONE DIRECTORY>/notebooks/hfdemo/tspulse/anomaly_detection 
$ python run_experiment.py --data_direc "Datasets/TSB-AD-U" --eval_file "Datasets/File_List/TSB-AD-U-Tuning.csv"  --mode "time" --out_file "benchmarks/TSB-AD-U-FT-Tuning-time.csv" --finetune --epochs 10 --decoder common_channel
$ python run_experiment.py --data_direc "Datasets/TSB-AD-U" --eval_file "Datasets/File_List/TSB-AD-U-Tuning.csv"  --mode "fft" --out_file "benchmarks/TSB-AD-U-FT-Tuning-fft.csv" --finetune --epochs 10 --decoder common_channel
$ python run_experiment.py --data_direc "Datasets/TSB-AD-U" --eval_file "Datasets/File_List/TSB-AD-U-Tuning.csv"  --mode "forecast" --out_file "benchmarks/TSB-AD-U-FT-Tuning-future.csv" --finetune --epochs 10 --decoder common_channel
$ python run_experiment.py --data_direc "Datasets/TSB-AD-U" --eval_file "Datasets/File_List/TSB-AD-U-Tuning.csv"  --mode "time+fft+forecast" --out_file "benchmarks/TSB-AD-U-FT-Tuning-ebsemble.csv" --finetune --epochs 10 --decoder common_channel
$ python run_experiment.py --data_direc "Datasets/TSB-AD-U" --eval_file "Datasets/File_List/TSB-AD-U-Eva.csv"  --mode "time" --out_file "benchmarks/TSB-AD-U-FT-Eva-time.csv" --finetune --epochs 10 --decoder common_channel
$ python run_experiment.py --data_direc "Datasets/TSB-AD-U" --eval_file "Datasets/File_List/TSB-AD-U-Eva.csv"  --mode "fft" --out_file "benchmarks/TSB-AD-U-FT-Eva-fft.csv" --finetune --epochs 10 --decoder common_channel
$ python run_experiment.py --data_direc "Datasets/TSB-AD-U" --eval_file "Datasets/File_List/TSB-AD-U-Eva.csv"  --mode "forecast" --out_file "benchmarks/TSB-AD-U-FT-Eva-future.csv" --finetune --epochs 10 --decoder common_channel
$ python run_experiment.py --data_direc "Datasets/TSB-AD-U" --eval_file "Datasets/File_List/TSB-AD-U-Eva.csv"  --mode "time+fft+forecast" --out_file "benchmarks/TSB-AD-U-FT-Eva-ebsemble.csv" --finetune --epochs 10 --decoder common_channel
```

Similarly the finetuning results for multi-variate benchmarking can be generated by running the following scripts.
```bash
$ cd <PATH TO GRANITE-TSFM CLONE DIRECTORY>/notebooks/hfdemo/tspulse/anomaly_detection 
$ python run_experiment.py --data_direc "Datasets/TSB-AD-M" --eval_file "Datasets/File_List/TSB-AD-M-Tuning.csv"  --mode "time" --out_file "benchmarks/TSB-AD-M-FT-Tuning-time.csv" --finetune --epochs 10 --decoder common_channel
$ python run_experiment.py --data_direc "Datasets/TSB-AD-M" --eval_file "Datasets/File_List/TSB-AD-M-Tuning.csv"  --mode "fft" --out_file "benchmarks/TSB-AD-M-FT-Tuning-fft.csv" --finetune --epochs 10 --decoder common_channel
$ python run_experiment.py --data_direc "Datasets/TSB-AD-M" --eval_file "Datasets/File_List/TSB-AD-M-Tuning.csv"  --mode "forecast" --out_file "benchmarks/TSB-AD-M-FT-Tuning-future.csv" --finetune --epochs 10 --decoder common_channel
$ python run_experiment.py --data_direc "Datasets/TSB-AD-M" --eval_file "Datasets/File_List/TSB-AD-M-Tuning.csv"  --mode "time+fft+forecast" --out_file "benchmarks/TSB-AD-M-FT-Tuning-ebsemble.csv" --finetune --epochs 10 --decoder common_channel
$ python run_experiment.py --data_direc "Datasets/TSB-AD-M" --eval_file "Datasets/File_List/TSB-AD-M-Eva.csv"  --mode "time" --out_file "benchmarks/TSB-AD-M-FT-Eva-time.csv" --finetune --epochs 10 --decoder common_channel
$ python run_experiment.py --data_direc "Datasets/TSB-AD-M" --eval_file "Datasets/File_List/TSB-AD-M-Eva.csv"  --mode "fft" --out_file "benchmarks/TSB-AD-M-FT-Eva-fft.csv" --finetune --epochs 10 --decoder common_channel
$ python run_experiment.py --data_direc "Datasets/TSB-AD-M" --eval_file "Datasets/File_List/TSB-AD-M-Eva.csv"  --mode "forecast" --out_file "benchmarks/TSB-AD-M-FT-Eva-future.csv" --finetune --epochs 10 --decoder common_channel
$ python run_experiment.py --data_direc "Datasets/TSB-AD-M" --eval_file "Datasets/File_List/TSB-AD-M-Eva.csv"  --mode "time+fft+forecast" --out_file "benchmarks/TSB-AD-M-FT-Eva-ebsemble.csv" --finetune --epochs 10 --decoder common_channel
``` 

Zero-Shot triangulation results on univariate datasets can be reproduced using `triangulation_scoring.py` script by running the following command.
```bash
$ python triangulation_scoring.py --root_directory benchmarks  --prefix "TSB-AD-U-" --suffix ".csv" --metric "VUS-PR" --eval_prefix "Eva" --tuning_prefix "Tuning" 

Triangulated VUS-PR: 0.480
```

Finetuned triangulation results on univariate datasets can be reproduced by running the following command.
```bash
$ python triangulation_scoring.py --root_directory benchmarks  --prefix "TSB-AD-U-FT-" --suffix ".csv" --metric "VUS-PR" --eval_prefix "Eva" --tuning_prefix "Tuning"

Triangulated VUS-PR: 0.558
```

Zero-Shot triangulation results on multi-variate datasets can be reproduced by running the following command.
```bash
$ python triangulation_scoring.py --root_directory benchmarks  --prefix "TSB-AD-M-" --suffix ".csv" --metric "VUS-PR" --eval_prefix "Eva" --tuning_prefix "Tuning" 

Triangulated VUS-PR: 0.361
```

Finetuned triangulation results on multi-variate datasets can be reproduced by running following command.
```bash
$ python triangulation_scoring.py --root_directory benchmarks  --prefix "TSB-AD-M-FT-" --suffix ".csv" --metric "VUS-PR" --eval_prefix "Eva" --tuning_prefix "Tuning"

Triangulated VUS-PR: 0.387
```
