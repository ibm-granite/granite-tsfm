# 🚀 Setup Guide For TTM-R3 GIFT Result Reproducability

Run the following commands step-by-step:


# 1. Create environment
conda create --prefix ./env python=3.11 -y

# 2. Activate environment
conda activate ./env

# 3. Install core dependencies
pip install git+https://github.com/ibm-granite/granite-tsfm.git@v0.2.28
pip install torch==2.4.0 "transformers>=4.44.0,<4.51.0" openpyxl

# 4. Install Ray (if required)
pip install ray==2.54.0

# 5. Clone and install evaluation repo
git clone https://github.com/<your-username>/gift-eval
cd gift-eval
pip install -e .
cd ..

# 6. Clone Granite TSFM and checkout TTM-R3 branch
git clone https://github.com/ibm-granite/granite-tsfm
cd granite-tsfm
git checkout ttm-r3-release-mq2
cd ..

# 7. Set PYTHONPATH
export PYTHONPATH=$(pwd)/granite-tsfm

echo "✅ Setup Complete"

# =========================================================
# 🚀 Run Instructions
# =========================================================

cd notebooks/hfdemo/tinytimemixer/full_benchmarking/gift_leaderboard_ttm_r3_nc

# Zero-shot
sh scripts/run_ttm_r3_zs.sh

# Few-shot
sh scripts/run_ttm_r3_ft.sh

# Zero-shot Lite
sh scripts/run_ttm_r3_zs_lite.sh

# Few-shot Lite
sh scripts/run_ttm_r3_ft_lite.sh