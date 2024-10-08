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
	python -m tsfminference.main &
	sleep 10
stop_service_local:
	pkill  -f 'python.*tsfminference.*'
	sleep 10

image: boilerplate
	$(CONTAINER_BUILDER) build -t tsfminference -f Dockerfile .

start_service_image: image
	$(CONTAINER_BUILDER) run -p 8000:8000 -d --rm --name tsfmserver tsfminference
	sleep 10

start_service_image_no_hf: image
	$(CONTAINER_BUILDER) run -p 8000:8000 \
	  -e "TSFM_ALLOW_LOAD_FROM_HF_HUB=0" \
	  -e "TSFM_MODEL_DIR=/" \
	  -v ./ibm:/ibm \
	  --rm --name tsfmserver tsfminference
	sleep 10

stop_service_image:
	$(CONTAINER_BUILDER) stop tsfmserver

test_local: boilerplate start_service_local
	pytest tests
	$(MAKE) stop_service_local

test_image: start_service_image
	pytest tests
	$(MAKE) stop_service_image

install_dev: boilerplate
	pip install poetry && poetry install --with dev




