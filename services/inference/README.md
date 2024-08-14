# TSFM Services



This component provides RESTful services for the  tsfm-granite class of 
timeseries foundation models. At present it can serve the following models:

* https://huggingface.co/ibm-granite/granite-timeseries-ttm-v1
* https://huggingface.co/ibm-granite/granite-timeseries-patchtst
* https://huggingface.co/ibm-granite/granite-timeseries-patchtsmixer
  


## Prerequisites:

* GNU make
* python >=3.10, <3.13
* poetry (`pip install poetry`)
* zsh or bash

_Note that our primary target environment for services deployment is x86_64 Linux. 
You may encounter hiccups if you try to use this on a different environment. 
If so, please file an issue. Some of our developers do use a Mac so you're 
likely to find a quick resolution. None of our developers use native Windows, 
however._

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

### Creating an image

_You must have either docker or podman installed on your system for this to
work. You must also have proper permissions on your system to build images._

```sh
CONTAINER_BUILDER=<docker|podmain> make image
# e.g, CONTAINER_BUILDER=docker make image
```

After a successful build you should have a local image named 
`tsfminference:latest`

```sh
(py311) ➜  tsfm-services git:(revised-build-system) ✗ docker images | grep tsfminference | head -n 1
tsfminference                                             latest               df592dcb0533   46 seconds ago      1.49GB
# some of the numeric and hash values on your machine could be different
```

## Runing the service unit tests

### Using docker or podman

```sh
CONTAINER_BUILDER=<docker|podman> make test_image

docker run -p 8000:8000 -d --rm --name tsfmserver tsfminference
1f88368b7c133ce4a236f5dbe3be18a23e98f65871e822ad808cf4646106fc9e
sleep 10
pytest tests
================================ test session starts ===========================
platform linux -- Python 3.11.9, pytest-8.3.2, pluggy-1.5.0
rootdir: /home/stus/git/github.com/tsfm_public/services/inference
configfile: pyproject.toml
plugins: anyio-4.4.0
collected 3 items                                                                                                                                                                           
# this list of tests is illustrative only
tests/test_inference.py ...                                                                                                                                                           [100%]

====================================== 3 passed in 3.69s =======================
```

### Testing on a local kubernetes cluster using kind

For this example we'll use [kind](https://kind.sigs.k8s.io/docs/user/quick-start/),
a lightweight way of running a local kubernetes cluster using docker. Before 
proceding, please follow the kind 
[installation guide](https://kind.sigs.k8s.io/docs/user/quick-start/).

* Create a local cluster

```bash
kind create cluster
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
```

* Confirm that `kubectl` is using the local cluster as its context

```bash
kubectl config current-context 
kind-kind
```

* Install kserve inside your local cluster:

```bash
curl -s https://raw.githubusercontent.com/kserve/kserve/release-0.13/hack/quick_install.sh | bash
```

This might take a little while to complete because a number of kserve-related containers 
need to be pulled and started. The script may fail the first time around (
   you can run it multiple times without harm.
) because some containers might still be in the pull or starting state. 
You can confirm that all of kserve's containers have started properly by doing:

```bash
 kubectl get pods --all-namespaces

NAMESPACE            NAME                                         READY   STATUS    RESTARTS   AGE
cert-manager         cert-manager-b748b77ff-q88k5                 1/1     Running   0          3m45s
cert-manager         cert-manager-cainjector-7ccb6546c6-f4bgf     1/1     Running   0          3m45s
cert-manager         cert-manager-webhook-7f669bd79f-p76rv        1/1     Running   0          3m45s
istio-system         istio-ingressgateway-76c99fc86c-t5fnr        1/1     Running   0          7m12s
istio-system         istiod-7f7457d5fc-gzlxb                      1/1     Running   0          7m54s
knative-serving      activator-58db57894b-6dznh                   1/1     Running   0          6m46s
knative-serving      autoscaler-76f95fff78-h62tw                  1/1     Running   0          6m45s
knative-serving      controller-7dd875844b-bcf5r                  1/1     Running   0          6m45s
knative-serving      net-istio-controller-57486f879-6lx8w         1/1     Running   0          6m44s
knative-serving      net-istio-webhook-7ccdbcb557-vfwbp           1/1     Running   0          6m44s
knative-serving      webhook-d8674645d-qrb4j                      1/1     Running   0          6m45s
kserve               kserve-controller-manager-589c77df99-p9tw2   1/2     Running   0          2m
kube-system          coredns-76f75df574-257xj                     1/1     Running   0          15m
kube-system          coredns-76f75df574-szdnf                     1/1     Running   0          15m
kube-system          etcd-kind-control-plane                      1/1     Running   0          16m
kube-system          kindnet-qvmc5                                1/1     Running   0          15m
kube-system          kube-apiserver-kind-control-plane            1/1     Running   0          16m
kube-system          kube-controller-manager-kind-control-plane   1/1     Running   0          16m
kube-system          kube-proxy-8vl2j                             1/1     Running   0          15m
kube-system          kube-scheduler-kind-control-plane            1/1     Running   0          16m
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
