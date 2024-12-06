CONTAINER_BUILDER ?= docker

# copies boilerplate code to suitable locations
boilerplate:
	for f in ../boilerplate/*.py; do \
		echo $$f; \
		cat ../boilerplate/warning.txt > tsfmfinetuning/$$(basename $$f); \
		cat $$f>>tsfmfinetuning/$$(basename $$f); \
		done 
	
image:
	$(CONTAINER_BUILDER) build -t tsfmfinetuning -f Dockerfile .

clone_models:
	git lfs install || true
	git clone https://huggingface.co/ibm/test-tsfm mytest-tsfm || true


fetchdata:
	python tests/fetchdata.py

test_local: boilerplate fetchdata clone_models
	pytest -s tests

install_dev: boilerplate
	pip install poetry && poetry install --with dev
