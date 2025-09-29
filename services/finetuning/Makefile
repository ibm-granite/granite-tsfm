CONTAINER_BUILDER ?= docker
EXTRA_DOCKER_BUILD_ARGS ?= --format docker --network host
EXTRA_DOCKER_BUILD_ARGS := $(if $(filter docker,$(CONTAINER_BUILDER)),,$(EXTRA_DOCKER_BUILD_ARGS))


# copies boilerplate code to suitable locations
boilerplate:
	for f in ../boilerplate/*.py; do \
		echo $$f; \
		cat ../boilerplate/warning.txt > tsfmfinetuning/$$(basename $$f); \
		cat $$f>>tsfmfinetuning/$$(basename $$f); \
		if ! grep $$(basename $$f) tsfmfinetuning/.gitignore; then echo $$(basename $$f) >> tsfmfinetuning/.gitignore; fi \
		done

image: boilerplate
        # build our base inference images first
	@cd ../inference ; make image
	$(CONTAINER_BUILDER) \
	build $(EXTRA_DOCKER_BUILD_ARGS) -t tsfmfinetuning-gpu \
	--build-arg CODEDIR="tsfmfinetuning" \
	-f Dockerfile.gpu .

clone_models:
	rm -rf mytest-tsfm || true
	git lfs install || true
	git clone --depth 1 -b refactor_v2 https://huggingface.co/ibm-research/test-tsfm mytest-tsfm 


fetchdata:
	rm -rf data || true
	python tests/fetchdata.py

test_local: clone_models fetchdata boilerplate
	pytest -s tests

install_dev: boilerplate
	pip install poetry && poetry install --with dev
