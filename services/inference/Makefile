CONTAINER_BUILDER ?= docker

# starts the inference service (used mainly for test cases)
start_service_local:
	python -m tsfmservices.inference.main &
	sleep 10
stop_service_local:
	pkill  -f 'python.*tsfmservices.*'
	sleep 10

image:
	$(CONTAINER_BUILDER) build -t tsfmservices -f Dockerfile .

start_service_image: image
	$(CONTAINER_BUILDER) run -p 8000:8000 -d --rm --name tsfmserver tsfmservices
	sleep 10
stop_service_image:
	$(CONTAINER_BUILDER) stop tsfmserver

test_local: start_service_local
	pytest tests
	$(MAKE) stop_service_local

test_image: start_service_image
	pytest tests
	$(MAKE) stop_service_image




