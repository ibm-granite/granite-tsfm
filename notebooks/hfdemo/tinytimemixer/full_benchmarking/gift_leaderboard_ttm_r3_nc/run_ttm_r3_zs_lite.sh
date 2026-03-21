source ~/.bashrc

# Change this according to your environment
conda activate gift

export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"

# Do not change this
export TTM_MODEL_SOURCE=public

python  ttm_r3_RAY.py  -ubfs 1 \
                    -tv TTM-R3-Pretrained-Lite \
                    -as 1 \
                    -ne 0 \
                    -rn 1 \
                    -fze 1 \
                    -fze_mode backtest_mean \
                    -fze_ratio 0.7 \
                    -mmp resources/map.json \
                    -mdp resource/ttm_r3_paths.json \
                    --use_lite 1
                    
                            