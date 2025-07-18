TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun ttm_pretrain_decompose_sample.py  --context_length 512 \
                               --forecast_length 96 \
                               --patch_length 8 \
                               --batch_size 64 \
                               --num_layers 5 \
                               --decoder_num_layers 3 \
                               --dropout 0.7 \
                               --head_dropout 0.7 \
                               --early_stopping 1 \
                               --adaptive_patching_levels 0 \
                               --num_epochs 5 \
                               --multi_scale 1 \
                               --register_tokens 5 \
                               --fft_length 10 \
                               --patch_gating 0 \
                               --scaling revin \
                               --multi_scale_loss 0 \
                               --use_fft_embedding 1 \
                               --enable_fourier_attention 0 \
                               --decoder_mode common_channel \
                               --trend_patch_length 64 \
                               --trend_patch_stride 64 \
                               --trend_d_model 64 \
                               --trend_decoder_d_model 64 \
                               --trend_num_layers 3 \
                               --trend_decoder_num_layers 2 \
                               --save_dir .\
                               



