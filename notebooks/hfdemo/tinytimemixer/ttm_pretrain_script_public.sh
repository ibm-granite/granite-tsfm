export PYTHONPATH=/dccstor/tsfm-irl/vijaye12/opensource/granite-tsfm:/dccstor/tsfm-irl/vijaye12/hf/tsfm:$PYTHONPATH

python ttm_pretrain_sample.py  --context_length 90 \
                               --forecast_length 30 \
                               --patch_length 10 \
                               --batch_size 64 \
                               --num_layers 3 \
                               --decoder_num_layers 3 \
                               --dropout 0.2 \
                               --head_dropout 0.2 \
                               --early_stopping 1 \
                               --adaptive_patching_levels 0 \
                               --num_epochs 1 \
                               --register_tokens 2 \
                               --fft_length 2 \
                               --save_dir /dccstor/tsfm-irl/vijaye12/opensource/g_fig/public
