# Steps to run the full benchmarking

1. In terminal, the any one of the three bash scripts `granite-r2.sh`, `granite-r1.sh`, or `research-use-r2.sh`.
2. Run `summarize_results.py`. For example, 
```
sh granite-r2.sh
python summarize_results.py -rd=results-granite-r2/
```

It will run all benchmarking and dump the results. The dumped results are available in the CSV files. 
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