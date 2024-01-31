# Adapted from HF Transformers: https://github.com/huggingface/transformers/tree/main
.PHONY: quality style

check_dirs := tests src tsfm notebooks


# this target runs checks on all files

quality:
	ruff check $(check_dirs) setup.py
	ruff format --check $(check_dirs) setup.py

# this target runs checks on all files and potentially modifies some of them

style:
	ruff check $(check_dirs) setup.py --fix 
	ruff format $(check_dirs) setup.py

