cl=$1
# for cl in 512 1024 1536; do 
for fl in 96 192 336 720; do
    python notebooks/hfdemo/tinytimemixer/full_benchmarking/ttm_full_benchmarking_sh.py --context_length $cl --forecast_length $fl --num_epochs 50 --num_workers 16 \
    --hf_model_path ibm-granite/granite-timeseries-ttm-r2 \
    --data_root_path $data_root_path \
    --save_dir results-granite-r1/
done;
# done;
