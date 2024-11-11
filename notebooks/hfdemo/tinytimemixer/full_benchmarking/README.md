# Steps to run the full benchmarking

## Fetching the data
The evaluation data can be downloaded from any of the previous time-series github repos like autoformer or timesnet or informer. [Sample download link](https://drive.google.com/drive/folders/1vE0ONyqPlym2JaaAoEe0XNDR8FS_d322). The ETT datasets can also be downloaded from [ETT-Github-Repository](https://github.com/zhouhaoyi/ETDataset).

Download and save the datasets in a directory. For example, in `data_root_path`. 
CSVs of each data should reside in location `data_root_path/$dataset_name/$dataset_name.csv` for our data utils to process them automatically.

## Running the scripts

1. In terminal, the any one of the three bash scripts `granite-r2.sh`, `granite-r1.sh`, or `research-use-r2.sh`.
2. Run `summarize_results.py`. For example, 
```
sh granite-r2.sh data_root_path/
python summarize_results.py -rd=results-granite-r2/
```

It will run all benchmarking and dump the results. The dumped results are available in the CSV files. 


## Benchmarking Results 
Note that, although random seed has been set, the mean squared error (MSE) scores might not match the below scores exactly depending on the runtime environment. The following results were obtained in a Unix-based machine equipped with one NVIDIA A-100 GPU.

1. TTM-Research-Use model results:
    - `combined_results-research-use-r2.csv`: Across all datasets, all TTM models, and all forecast horizons.
    - `combined_avg_results-research-use-r2.csv`: Across all datasets and all TTM models average over forecast horizons.
2. TTM-Granite-R2 model results:
    - `combined_results-granite-r2.csv`: Across all datasets, all TTM models, and all forecast horizons.
    - `combined_avg_results-granite-r2.csv`: Across all datasets and all TTM models average over forecast horizons.
2. TTM-Granite-R1 model results:
    - `combined_results-granite-r1.csv`: Across all datasets, all TTM models, and all forecast horizons.
    - `combined_avg_results-granite-r1.csv`: Across all datasets and all TTM models average over forecast horizons.  
    Note that TTM-Granite-R1 models supports 512/1024 as context length, and 96 as the forecast horizon.

# Sample benchmarking notebooks
We also provide a bunch of sample benchmarking notebooks in the `sample_notebooks` folder. These notebooks can be directly run or modifed according to the need. 