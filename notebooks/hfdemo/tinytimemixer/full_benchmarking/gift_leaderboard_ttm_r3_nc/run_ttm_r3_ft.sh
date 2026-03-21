source ~/.bashrc

# Change this according to your environment
conda activate gift

export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"

# Do not change this
export TTM_MODEL_SOURCE=public

python  ttm_r3_RAY.py  -ubfs 1 \
                        -tv TTM-R3-Finetuned \
                        -as 1 \
                        -ne 5 \
                        -dt 0  \
                        -bt 1 \
                        -aff 1 \
                        -ht 0 \
                        -prt 0 \
                        -pp 0 \
                        -qt 0 \
                        -rn 1 \
                        -fze 1 \
                        -fze_mode backtest_mean \
                        -fze_ratio 0.7 \
                        -fsdlc resources/fewshot_data_limit.json
                    
                    