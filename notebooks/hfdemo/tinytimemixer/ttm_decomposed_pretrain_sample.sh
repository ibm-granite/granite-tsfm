export PYTHONPATH=/dccstor/tsfm-irl/vijaye12/opensource/granite-tsfm:/dccstor/tsfm-irl/vijaye12/hf/tsfm:$PYTHONPATH

python ttm_decomposed_pretrain_sample.py  \
  --save_dir ./ttm_runs \
  --dataset_root_path /dccstor/tsfm23/datasets \
  --num_epochs 9

