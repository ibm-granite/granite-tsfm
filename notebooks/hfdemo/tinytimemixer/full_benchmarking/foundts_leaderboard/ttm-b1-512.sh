data_root_path=$1
for cl in 512; do 
    for fl in 96 192 336 720; do
        python ../ttm_full_benchmarking.py --context_length $cl --forecast_length $fl --num_epochs 50 --num_workers 16 \
        --hf_model_path ibm-research/ttm-research-r2  \
        --data_root_path $data_root_path \
        --fewshot 0 \
        --plot 0 \
        --enable_prefix_tuning 1 \
        --datasets etth1,etth2,ettm1,ettm2,weather,electricity,traffic,exchange,zafnoo \
        --save_dir results-ttm-b1/
    done;
done;
