# TSFM Inference Services

The TSFM Inference Services component provides a runtime for inference related tasks for the tsfm-granite class of 
timeseries foundation models. At present it provide inference endpoints for the following models:

* https://huggingface.co/ibm-granite/granite-timeseries-ttm-v1
* https://huggingface.co/ibm-granite/granite-timeseries-patchtst
* https://huggingface.co/ibm-granite/granite-timeseries-patchtsmixer
  
## Limitations

* At present the API includes only a forecasting endpoint. Other task-based endpoints such as regression and classification are in the works.
* The primary target environment is x86_64 Linux. 
You may encounter hiccups if you try to use this on a different environment. If so, please file an issue. Many of our developers do use a Mac and run all tests not involving building containers locally so you're likely to find a quick resolution. None of our developers use native Windows, however, and we do not plan on supporting that environment. Windows users are advised to use Microsoft's excellent WSL2 implementation which provides a native Linux environment running under Windows without the overheads of virtualization.


## Prerequisites:

* GNU make
* python >=3.10, <3.13
* poetry (`pip install poetry`)
* zsh or bash
* (optional) docker or podman
* (optional) kubectl if you plan on testing kubernetes-based deployments

## Known issues:

* Use of pkill statements in Makefile may not work properly on Mac OS. This will
 be apparent if you have left over processs after running test related make 
 targets. Please help us put OS-specific checks into our Makefile to handle 
 these cases by filing a PR.

## Installation

```sh
pip install poetry && poetry install --with dev
```

### Testing using a local server instance

```sh
make test_local
```

### Building an image

You must have either docker or podman installed on your system for this to
work. You must also have proper permissions on your system to build images. We assume you have a working docker command which can be docker itself 
or `podman` that has been aliased as `docker` or has been installed with the podman-docker package that will do this for you.

```bash
make image
```

After a successful build you should have a local image named 
`tsfminference:latest`

```sh
docker images | grep tsfminference | head -n 1
tsfminference                                             latest               df592dcb0533   46 seconds ago      1.49GB
# some of the numeric and hash values on your machine could be different
```

## Running the service unit tests

### Using the built image

```sh
make test_image

docker run -p 8000:8000 -d --rm --name tsfmserver tsfminference
1f88368b7c133ce4a236f5dbe3be18a23e98f65871e822ad808cf4646106fc9e
sleep 10
pytest tests
================================ test session starts ===========================
platform linux -- Python 3.11.9, pytest-8.3.2, pluggy-1.5.0
rootdir: /home/stus/git/github.com/tsfm_public/services/inference
configfile: pyproject.toml
plugins: anyio-4.4.0
collected 3 items                                                                                                         # this list of tests is illustrative only
tests/test_inference.py ...                                                                                                                                                           [100%]

====================================== 3 passed in 3.69s =======================
```

## Testing on a local kubernetes cluster using kind

For this example we'll use [kind](https://kind.sigs.k8s.io/docs/user/quick-start/),
a lightweight way of running a local kubernetes cluster using docker.

### Create a local cluster

First:

* [Install kubectl](https://kubernetes.io/docs/tasks/tools/)
* [Install kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation)
* [Install helm](https://helm.sh/docs/intro/install/)
* If you are using podman, you will need to enable the use of an insecure (using http instead of https)
local container registry by creating a file called `/etc/containers/registries.conf.d/localhost.conf` 
with the following content:

  ```
  [[registry]]
  location = "localhost:5001"
  insecure = true
  ```
* If you're using podman, you may run into issues running the kserve container due to 
open file (nofile) limits. If so, 
see https://github.com/containers/common/blob/main/docs/containers.conf.5.md
for instructions on how to increase the default limits.

Now install a kind control plane with a local docker registry:

```bash
curl -s https://kind.sigs.k8s.io/examples/kind-with-registry.sh | bash

Creating cluster "kind" ...
 ✓ Ensuring node image (kindest/node:v1.29.2) 🖼
 ✓ Preparing nodes 📦  
 ✓ Writing configuration 📜 
 ✓ Starting control-plane 🕹️ 
 ✓ Installing CNI 🔌 
 ✓ Installing StorageClass 💾 
Set kubectl context to "kind-kind"
You can now use your cluster with:

kubectl cluster-info --context kind-kind

Have a nice day! 👋
configmap/local-registry-hosting created
```

### Upload the tsfm service image to the kind local registry:

```bash
# don't forget to run "make image" first
docker tag tsfminference:latest localhost:5001/tsfminference:latest
  docker push localhost:5001/tsfminference:latest
```

Confirm that `kubectl` is using the local cluster as its context

```bash
kubectl config current-context 
kind-kind
```

### Install kserve inside your local cluster:

```bash
curl -s https://raw.githubusercontent.com/kserve/kserve/master/hack/quick_install.sh > quick_install.sh
bash ./quick_install.sh -r
```

This will take a minute or so to complete because a number of kserve-related containers 
need to be pulled and started. The script may fail the first time around (you can run it multiple times without harm) because some containers might still be in the pull or starting state. You can confirm that all of kserve's containers have started properly by doing (you may see some additional cointainers listed that are part of kind's internal services).

You know you have a successful install when you finally see (it might take two 
or more tries):

```bash
# [snip]...
clusterstoragecontainer.serving.kserve.io/default created
😀 Successfully installed KServe
```
### Deploy the tsfm kserve service

Save the folling yaml snippet to a file called tsfm.yaml.

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  # We're using a RawDeployment for testing purposes
  # as well as to get around an issue with the knative
  # operator reading from a local container registry
  annotations:
    serving.kserve.io/deploymentMode: RawDeployment
  name: tsfminferenceserver
spec:
  predictor:
    containers:
      - name: "tsfm-server"
        image: "localhost:5001/tsfminference:latest"
        imagePullPolicy: Always
        ports:
          - containerPort: 8000
            protocol: TCP
```

```bash
kubectl apply -f tsfm.yaml
```

Confirm that the service is running:

```bash
kubectl get inferenceservices.serving.kserve.io tsfminferenceserver 
NAME                  URL                                                  READY   PREV   LATEST   PREVROLLEDOUTREVISION   LATESTREADYREVISION   AGE
tsfminferenceserver   http://tsfminferenceserver-kserve-test.example.com   True                                                                  25m

```

Create a port forward for the predictor pod:

```bash
# your pod identifier suffix will be different
kubectl port-forward pods/tsfminferenceserver-predictor-7dcd6ff5d5-8f726 8000:8000   
```

Run the unit tests:

```bash
pytest tests
```

## Viewing the OpenAPI 3.x specification and swagger page

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
