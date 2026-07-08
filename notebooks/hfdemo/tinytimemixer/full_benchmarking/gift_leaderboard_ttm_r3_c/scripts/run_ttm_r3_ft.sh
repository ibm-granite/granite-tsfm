REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
source ${REPO_ROOT}/.venv/bin/activate
export PYTHONPATH="${REPO_ROOT}"
cd ..
python  ttm_r3_RAY.py \
        -ubfs 1 \
        -tv TTM-R3-FT-COM \
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
        -fsdlc ../gift_leaderboard_ttm_r3_nc/resources/fewshot_data_limit.json
    
    