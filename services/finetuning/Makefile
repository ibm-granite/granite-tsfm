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

test_local:
	pytest tests

install_dev: boilerplate
	pip install poetry && poetry install --with dev
