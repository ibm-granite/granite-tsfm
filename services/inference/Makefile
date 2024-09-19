CONTAINER_BUILDER ?= docker

# copies boilerplate code to suitable locations
boilerplate:
	cp ../boilerplate/warning.txt tsfminference/inference_payloads.py
	cat ../boilerplate/inference_payloads.py >> tsfminference/inference_payloads.py
	
# starts the tsfminference service (used mainly for test cases)
start_service_local:
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
stop_service_image:
	$(CONTAINER_BUILDER) stop tsfmserver

test_local: boilerplate start_service_local
	pytest tests
	$(MAKE) stop_service_local

test_image: start_service_image
	pytest tests
	$(MAKE) stop_service_image




