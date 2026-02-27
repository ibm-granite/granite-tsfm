export PYTHONPATH=/dccstor/tsfm-irl/vijaye12/opensource/granite-tsfm:/dccstor/tsfm-irl/vijaye12/hf/tsfm:$PYTHONPATH

python ttm_decomposed_test.py  --context_length 1536 \
                               --forecast_length 512 \
                               --patch_length 16 \
                               --patch_stride 16 \
                               --batch_size 64 \
                               --num_layers 3 \
                               --d_model 32 \
                               --decoder_d_model 8 \
                               --decoder_num_layers 1 \
                               --dropout 0.7 \
                               --head_dropout 0.7 \
                               --early_stopping \
                               --adaptive_patching_levels 0 \
                               --num_epochs 30 \
                               --register_tokens 1 \
                               --fft_length 0 \
                               --scaling std \
                               --residual_context_length 1536 \
                               --trend_patch_length 96 \
                               --trend_patch_stride 96 \
                               --trend_d_model 32 \
                               --trend_decoder_d_model 32 \
                               --trend_num_layers 1 \
                               --trend_decoder_num_layers 1 \
                               --epochs_phase1 3 \
                               --epochs_phase2 3 \
                               --epochs_phase3 5 \
                               --multi_quantile_head \
                               --multi_scale \
                               --mq_use_decoder_pool \
                               --mq_cond_path pool \
                               --mq_cond_mode add \
                               --mq_decoder_d_model 4 \
                               --mq_q50_type median \
                               --save_dir /dccstor/tsfm-irl/vijaye12/hacking/tmp_files/decom_style_multiscale \
                               
                               



