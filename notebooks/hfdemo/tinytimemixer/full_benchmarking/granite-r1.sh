data_root_path=$1
for cl in 512 1024; do 
    for fl in 96; do
        python ttm_full_benchmarking.py --context_length $cl --forecast_length $fl --num_epochs 50 --num_workers 16 \
        --hf_model_path ibm-granite/granite-timeseries-ttm-r1 \
        --data_root_path $data_root_path \
        --save_dir results-granite-r1/
    done;
done;
