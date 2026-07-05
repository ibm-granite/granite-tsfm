REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
source ${REPO_ROOT}/.venv/bin/activate
export PYTHONPATH="${REPO_ROOT}"
cd ..
python  ttm_r3_RAY.py \
    -ubfs 1 \
    -tv TTM-R3-PT-Lite-COM \
    -as 1 \
    -ne 0 \
    -rn 1 \
    -fze 1 \
    -fze_mode backtest_mean \
    -fze_ratio 0.7 \
    --use_lite 1
    
            