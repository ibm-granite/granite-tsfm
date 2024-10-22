CONTAINER_BUILDER ?= docker

# copies boilerplate code to suitable locations
boilerplate:
	for f in ../boilerplate/*.py; do \
		echo $$f; \
		cat ../boilerplate/warning.txt > tsfminference/$$(basename $$f); \
		cat $$f>>tsfminference/$$(basename $$f); \
		done 
	
# starts the tsfminference service (used mainly for test cases)
start_service_local: boilerplate
	python -m gunicorn \
	-w 1 \
	-k uvicorn.workers.UvicornWorker \
	--bind 127.0.0.1:8000 \
	tsfminference.main:app && true &
stop_service_local:
	pkill  -f 'python.*tsfminference.*'

image: boilerplate
	$(CONTAINER_BUILDER) build -t tsfminference -f Dockerfile .

start_service_image: clone_models image
	$(CONTAINER_BUILDER) run -p 8000:8000 \
	-d --rm --name tsfmserver \
	-e TSFM_MODEL_DIR=/tsfm_models \
	-e TSFM_ALLOW_LOAD_FROM_HF_HUB=0 \
	-v mytest-tsfm:/tsfm_models \
	tsfminference
	sleep 5

stop_service_image:
	$(CONTAINER_BUILDER) stop tsfmserver

test_local: clone_models boilerplate start_service_local
	TSFM_ALLOW_LOAD_FROM_HF_HUB=0 TSFM_MODEL_DIR=./mytest-tsfm \
	 pytest tests
	$(MAKE) stop_service_local
	$(MAKE) delete_models

test_image: start_service_image
	pytest tests
	$(MAKE) stop_service_image
	$(MAKE) delete_models

install_dev: boilerplate
	pip install poetry && poetry install --with dev

clone_models:
	git lfs install || true
	git clone https://huggingface.co/ibm/test-tsfm mytest-tsfm || true

delete_models:
	rm -rf mytest-tsfm || true



