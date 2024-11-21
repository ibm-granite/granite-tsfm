CONTAINER_BUILDER ?= docker

# copies boilerplate code to suitable locations
boilerplate:
	for f in ../boilerplate/*.py; do \
		echo $$f; \
		cat ../boilerplate/warning.txt > tsfminference/$$(basename $$f); \
		cat $$f>>tsfminference/$$(basename $$f); \
		done 

create_prometheus_metrics_dir:
	rm -rf prometheus_metrics > /dev/null 2>&1 | true
	mkdir -m go+rwX prometheus_metrics

# starts the tsfminference service (used mainly for test cases)
start_service_local: create_prometheus_metrics_dir boilerplate
	PROMETHEUS_MULTIPROC_DIR=./prometheus_metrics \
	TSFM_PYTHON_LOGGING_LEVEL="ERROR" \
	TSFM_MODEL_DIR=./mytest-tsfm \
	TSFM_ALLOW_LOAD_FROM_HF_HUB=1 \
	python -m gunicorn \
	-w 1 \
	-k uvicorn.workers.UvicornWorker \
	--bind 127.0.0.1:8000 \
	tsfminference.main:app && true &
stop_service_local:
	pkill  -f 'python.*gunicorn.*tsfminference\.main\:app'

image: boilerplate
	$(CONTAINER_BUILDER) build -t tsfminference -f Dockerfile .

start_service_image: create_prometheus_metrics_dir image
	$(CONTAINER_BUILDER) run -p 8000:8000 \
		-d \
		--rm \
		--name tsfmserver \
        -e TSFM_MODEL_DIR=/mytest-tsfm \
        -e TSFM_ALLOW_LOAD_FROM_HF_HUB=1 \
		-e PROMETHEUS_MULTIPROC_DIR=/prometheus_metrics \
        -v ./mytest-tsfm:/mytest-tsfm \
		-v ./prometheus_metrics:/prometheus_metrics \
        tsfminference
	sleep 5

stop_service_image:
	$(CONTAINER_BUILDER) stop tsfmserver

test_local: clone_models boilerplate start_service_local
	pytest --cov=tsfminference --cov-report term-missing tests ../tests
	$(MAKE) stop_service_local
	$(MAKE) delete_models
	$(MAKE) stop_service_local

test_image: clone_models start_service_image
	pytest -s tests ../tests
	$(MAKE) stop_service_image
	$(MAKE) delete_models

install_dev: boilerplate
	pip install poetry && poetry install --with dev

clone_models:
	git lfs install || true
	git clone https://huggingface.co/ibm/test-tsfm mytest-tsfm || true
	git clone https://huggingface.co/amazon/chronos-t5-small mytest-tsfm/chronos-t5-small || true
	cp tsfm_config.json mytest-tsfm/chronos-t5-small || true

delete_models:
	rm -rf mytest-tsfm || true



