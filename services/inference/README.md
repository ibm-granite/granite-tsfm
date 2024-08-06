# TSFM Services



This component provides RESTful services for the  tsfm-granite class of timeseries foundation models. At present it can serve the following models:

* https://huggingface.co/ibm-granite/granite-timeseries-ttm-v1
* https://huggingface.co/ibm-granite/granite-timeseries-patchtst
* https://huggingface.co/ibm-granite/granite-timeseries-patchtsmixer
  


## Prerequisites:

* GNU make
* python 3.10 or 3.11
* poetry (`pip install poetry`)
* zsh or bash (zsh is preferred)

_Note that our primary target environment for services deployment is x86_64 Linux. You may encounter hiccups if you try to use this on a different environment. If so, please
file an issue. Some of our developers do use a Mac so you're likely to find a quick resolution. None of our developers use native Windows, however._

## Known issues:

* Use of pkill statements in Makefile may not work properly on Mac OS. This will be apparent if you have left over processs after running test related make targets. Please help us put OS-specific checks into our Makefile to handle these cases by filing a PR.

### Installation

```sh
pip install poetry && poetry install --with dev
```

### Testing using a local server instance

```sh
make test_local
```

### Creating an image

_You must have either docker or podman installed on your system for this to
work. You must also have proper permissions on your system to build images._

```sh
CONTAINER_BUILDER=<docker|podmain> make image
# e.g, CONTAINER_BUILDER=docker make image
```

After a successful build you should have a local image named `tsfminference:latest`

```sh
(py311) ➜  tsfm-services git:(revised-build-system) ✗ docker images | grep tsfminference | head -n 1
tsfminference                                             latest               df592dcb0533   46 seconds ago      1.49GB
# some of the numeric and hash values on your machine could be different
```

### Testing using the built image

```sh
make test_image
```

### Viewing the OpenAPI 3.x specification and swagger page

```sh
make start_local_server
```

Then open your browser to http://127.0.0.1:8000

To stop the server run:

```sh
# may not work properly on a Mac
# if not, kill the uvicorn process manually.
make stop_local_server
```
