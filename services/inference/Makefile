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

start_service_image: image
	$(CONTAINER_BUILDER) run -p 8000:8000 -d --rm --name tsfmserver tsfminference
	sleep 5

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




