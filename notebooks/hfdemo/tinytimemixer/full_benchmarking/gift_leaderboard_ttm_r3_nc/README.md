## 🚀 Setup Guide for TTM-R3 GIFT Result Reproducibility

Follow the steps below to set up the environment and reproduce results.

---

## 🧱 1. Create Environment
```bash
conda create --prefix ./env python=3.11 -y
````

## ▶️ 2. Activate Environment

```bash
conda activate ./env
```

---

## 📦 3. Install Core Dependencies

```bash
pip install git+https://github.com/ibm-granite/granite-tsfm.git@v0.2.28
pip install torch==2.4.0 "transformers>=4.44.0,<4.51.0" openpyxl
```

---

## ⚡ 4. Install Ray (Optional)

```bash
pip install ray==2.54.0
```

---

## 📥 5. Clone and Install Evaluation Repo

```bash
git clone https://github.com/<your-username>/gift-eval
cd gift-eval
pip install -e .
cd ..
```

---

## 🧠 6. Clone Granite TSFM and Checkout TTM-R3 Branch

```bash
git clone https://github.com/ibm-granite/granite-tsfm
cd granite-tsfm
git checkout ttm-r3-release-mq2
cd ..
```

---

## 🔧 7. Set PYTHONPATH

```bash
export PYTHONPATH=$(pwd)/granite-tsfm
echo "✅ Setup Complete"
```

---

## 🚀 8. Execute Experiments

```bash
cd notebooks/hfdemo/tinytimemixer/full_benchmarking/gift_leaderboard_ttm_r3_nc
```

### 🔹 Zero-Shot

```bash
sh scripts/run_ttm_r3_zs.sh
```

### 🔹 Few-Shot

```bash
sh scripts/run_ttm_r3_ft.sh
```

### 🔹 Zero-Shot (Lite)

```bash
sh scripts/run_ttm_r3_zs_lite.sh
```

### 🔹 Few-Shot (Lite)

```bash
sh scripts/run_ttm_r3_ft_lite.sh
```

---

## 📓 Notebook Execution

Move the `ttm-r3.ipynb` notebook from gift-eval repo to this benchmarking directory and execute it.



