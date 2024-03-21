# Adapted from HF Transformers: https://github.com/huggingface/transformers/tree/main
.PHONY: quality style

check_dirs := tests tsfm_public tsfmhfdemos notebooks


# this target runs checks on all files

quality:
	ruff check $(check_dirs)
	ruff format --check $(check_dirs)

# this target runs checks on all files and potentially modifies some of them

style:
	ruff check $(check_dirs) --fix 
	ruff format $(check_dirs)

