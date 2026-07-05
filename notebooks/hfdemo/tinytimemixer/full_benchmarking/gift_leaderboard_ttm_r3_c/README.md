## 🚀 Setup Guide for TTM-R3 GIFT Result Reproducibility  

Follow the steps below to set up the environment and reproduce results.  
The following steps use `uv` package manager: https://docs.astral.sh/uv/getting-started/installation/ 

## 🧠 1. Clone Granite TSFM and Checkout TTM-R3 Branch

```bash
git clone https://github.com/ibm-granite/granite-tsfm
cd granite-tsfm
git fetch 
git switch ttm-r3-rel-com
```

---

## 🧱 2. Create Environment
```bash
uv venv .venv
````

---

## ▶️ 3. Activate Environment

```bash
source .venv/bin/activate
```

---

## 📦 4. Install Core Dependencies

```bash
uv pip install git+https://github.com/ibm-granite/granite-tsfm.git@v0.3.6
uv pip install openpyxl
```

---

## ⚡ 5. Install Ray (Optional)

```bash
uv pip install ray==2.54.0
```

---

## 📥 6. Clone and Install GIFT Evaluation Repo

```bash
cd ..
git clone https://github.com/SalesforceAIResearch/gift-eval.git
cd gift-eval
uv pip install -e .
cd ..
```

---

## 🔧 7. Set PYTHONPATH

```bash
export PYTHONPATH=$(pwd)/granite-tsfm
echo "✅ Setup Complete"
```

---

## 🚀 8. Execute Full Experiments

```bash
cd granite-tsfm/notebooks/hfdemo/tinytimemixer/full_benchmarking/gift_leaderboard_ttm_r3_c/scripts
```

### 🔹 Zero-Shot

```bash
bash run_ttm_r3_zs.sh
```

### 🔹 Few-Shot

```bash
bash run_ttm_r3_ft.sh
```

### 🔹 Zero-Shot (Lite)

```bash
bash run_ttm_r3_zs_lite.sh
```

### 🔹 Few-Shot (Lite)

```bash
bash run_ttm_r3_ft_lite.sh
```

---

## 📓 Notebook Execution

Run `ttm_r3.ipynb`.



