# Adapted from HF Transformers: https://github.com/huggingface/transformers/tree/main
.PHONY: quality style

check_dirs := tests tsfm_public tsfmhfdemos notebooks services


# this target runs checks on all files

quality:
	uv run ruff check $(check_dirs)
	uv run ruff format --check $(check_dirs)

# this target runs checks on all files and potentially modifies some of them

style:
	uv run ruff check $(check_dirs) --fix 
	uv run ruff format $(check_dirs)

# update uv lock files
# for cve compliance
update_lock_files:
	touch ./uv.lock services/finetuning/uv.lock services/inference/uv.lock
	rm ./uv.lock services/finetuning/uv.lock services/inference/uv.lock 
	uv lock -U
	uv lock --directory services/inference -U
	uv lock --directory services/finetuning -U
	uv run --directory services/inference make update_examples

