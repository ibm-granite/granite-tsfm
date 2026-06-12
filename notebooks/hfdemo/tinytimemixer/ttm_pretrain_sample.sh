export PYTHONPATH=/dccstor/tsfm-irl/vijaye12/opensource/granite-tsfm:/dccstor/tsfm-irl/vijaye12/hf/tsfm:$PYTHONPATH

python ttm_pretrain_sample.py \
  --batch_size 64 \
  --num_epochs 5 \
  --learning_rate 1e-4 \
  --num_workers 4 \
  --random_seed 42 \
  --early_stopping \
  --save_dir ./ttm_dirs