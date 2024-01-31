# Adapted from HF Transformers: https://github.com/huggingface/transformers/tree/main


check_dirs := tests tsfm_public notebooks tsfmhfdemos


style:
	ruff check $(check_dirs) setup.py --fix 
	ruff format $(check_dirs) setup.py
#	${MAKE} extra_style_checks

