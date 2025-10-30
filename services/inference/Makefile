CONTAINER_BUILDER ?= docker
SKIP_GPU_BUILD ?= "0"
NO_DOCKER_CACHE ?=
EXTRA_DOCKER_BUILD_ARGS ?= --format docker --network host
EXTRA_DOCKER_BUILD_ARGS := $(if $(filter docker,$(CONTAINER_BUILDER)),,$(EXTRA_DOCKER_BUILD_ARGS))

# copies boilerplate code to suitable locations
boilerplate:
	for f in ../boilerplate/*.py; do \
		echo $$f; \
		cat ../boilerplate/warning.txt > tsfminference/$$(basename $$f); \
		cat $$f>>tsfminference/$$(basename $$f); \
		if ! grep $$(basename $$f) tsfminference/.gitignore; then echo $$(basename $$f) >> tsfminference/.gitignore; fi \
		done

create_prometheus_metrics_dir:
	rm -rf prometheus_metrics > /dev/null 2>&1 | true
	mkdir -m go+rwX prometheus_metrics

# starts the tsfminference service (used mainly for test cases)
start_service_local_watch: create_prometheus_metrics_dir boilerplate
	PROMETHEUS_MULTIPROC_DIR=./prometheus_metrics \
	TSFM_PYTHON_LOGGING_LEVEL="ERROR" \
	TSFM_MODEL_DIR=./foobaz:./mytest-tsfm \
	TSFM_ALLOW_LOAD_FROM_HF_HUB=1 \
	uv run python -m gunicorn \
	-w 1 \
	--reload \
	-k uvicorn.workers.UvicornWorker \
	--bind 127.0.0.1:8000 \
	tsfminference.main:app
start_service_local:
	@make start_service_local_watch &
stop_service_local:
	pkill  -f '.*python.*gunicorn.*tsfminference\.main\:app' || true

image: boilerplate
	find . -name "*.pyc" | xargs rm -f || true
	$(CONTAINER_BUILDER) build $(EXTRA_DOCKER_BUILD_ARGS) -t tsfminference-cpu --build-arg CODEDIR="tsfminference" -f Dockerfile.cpu .
	@if [ "$$SKIP_GPU_BUILD" != "1" ]; then \
		$(CONTAINER_BUILDER) build $(EXTRA_DOCKER_BUILD_ARGS) -t tsfminference-gpu --build-arg CODEDIR="tsfminference" -f Dockerfile.gpu .; \
	else \
		echo "Skipping GPU build"; \
	fi

# it's very important that the below start statement
# NOT use uv b/c we're emulating downstream cosumers
# who might not deploy with that command
start_service_image: create_prometheus_metrics_dir image
	$(CONTAINER_BUILDER) run -p 8000:8000 \
		-d \
		--rm \
		--name tsfmserver-cpu \
        -e TSFM_MODEL_DIR=/mytest-tsfm \
        -e TSFM_ALLOW_LOAD_FROM_HF_HUB=1 \
		-e PROMETHEUS_MULTIPROC_DIR=/prometheus_metrics \
        -v ./mytest-tsfm:/mytest-tsfm \
		-v ./prometheus_metrics:/prometheus_metrics \
        tsfminference-cpu python \
		 -m gunicorn -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 tsfminference.main:app
	sleep 5

stop_service_image:
	$(CONTAINER_BUILDER) stop tsfmserver-cpu

test_local: clone_models boilerplate stop_service_local start_service_local
	uv run locust -f tests/locust/locustfile.py --config tests/locust/locust.conf \
	  --headless --only-summary --check-fail-ratio 0.01
	uv run pytest -s --cov=tsfminference --cov-report term-missing tests ../tests
	$(MAKE) stop_service_local

test_image: clone_models start_service_image
	uv run locust -f tests/locust/locustfile.py --config tests/locust/locust.conf \
	 --headless --only-summary --check-fail-ratio 0.01
	uv run pytest -s tests ../tests
	$(MAKE) stop_service_image

install_dev: boilerplate
	uv sync --locked --extra dev --editable

clone_models:
	git lfs install || true
	git clone -b refactor_v2 https://huggingface.co/ibm-research/test-tsfm mytest-tsfm || true

# update any auto-generated files
# used in example code
update_examples:
# our sagemaker stuff
	echo "# This file was automatically generated on $(shell date), your edits will be replaced." > examples/aws/sagemaker/tsfm_custom/code/requirements.txt
	uv export --omit-hashes >> examples/aws/sagemaker/tsfm_custom/code/requirements.txt

profile_inference: clone_models
	TSFM_TESTS_AS_PROFILER=1 \
	TSFM_PROFILE_NUM_TIMESERIES=10000 \
	uv run python -m cProfile -o profile.out -s cumtime \
	 -m pytest -s tests/test_inference_lib.py::test_forecast_with_good_data
	gprof2dot -f pstats profile.out | dot -Tpng -o profile_tree.png
	echo "generated profile_tree.png"
