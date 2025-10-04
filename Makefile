# Makefile for publishing timecopilot-granite-tsfm to PyPI

# Python version to test with
PYTHON_VERSION = 3.13

# Import statement to test
IMPORT_STATEMENT = from tsfm_public import FlowStateForPrediction

clean:
	rm -rf *.egg-info
	rm -rf dist/

# Build the wheel and sdist
build:
	uv build -n

# Test importing from built wheel
test-wheel:
	uv run --isolated --no-project -p $(PYTHON_VERSION) --with dist/*.whl -- python -c "$(IMPORT_STATEMENT)"

# Test importing from built sdist
test-sdist:
	uv run --isolated --no-project -p $(PYTHON_VERSION) --with dist/*.tar.gz -- python -c "$(IMPORT_STATEMENT)"

# Publish the package to PyPI using trusted publishing
publish:
	uv publish

# Test importing from the installed package after publishing
after-publish-test:
	uv run --isolated --no-project -p $(PYTHON_VERSION) --with timecopilot-timesfm -- python -c "$(IMPORT_STATEMENT)"

# All steps
release: clean build test-wheel test-sdist publish after-publish-test

