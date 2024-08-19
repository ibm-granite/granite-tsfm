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
work. You must also have proper permissions on your system to build images. If you are using podman, please also alias the `podman` command to `docker` so that the code snippets that reference `docker` below will work._

```sh
# this defaults to using the docker command
# if you're using podman, don't forget to
# alias docker=podman first
make image
```

After a successful build you should have a local image named 
`tsfminference:latest`

```sh
(py311) ➜  tsfm-services git:(revised-build-system) ✗ docker images | grep tsfminference | head -n 1
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
a lightweight way of running a local kubernetes cluster using docker. Before 
proceding, please follow the kind 
[installation guide](https://kind.sigs.k8s.io/docs/user/quick-start/) but do not create a local cluster.

### Create a local cluster

For this example, we need to install kind with a
local image registry. Download the [installer script](https://kind.sigs.k8s.io/examples/kind-with-registry.sh):

```bash
curl -L https://kind.sigs.k8s.io/examples/kind-with-registry.sh > /tmp/kind-with-registry.sh
```

```bash
sh /tmp/kind-with-registry.sh
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

### Upload our tsfm service image to the kind local registry:

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
curl -s https://raw.githubusercontent.com/kserve/kserve/v0.13.1/hack/quick_install.sh | bash
```

This might take a little while to complete because a number of kserve-related containers 
need to be pulled and started. The script may fail the first time around (you can run it multiple times without harm) because some containers might still be in the pull or starting state. 
You can confirm that all of kserve's containers have started properly by doing (you may see some additional cointainers listed that are part of kind's internal services):

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

### Confirm that you can perform a sklearn inference run (optional)

This optional section is to help you confirm that your kserve setup is correct
to isolate any k8s issues from TSFM-specific ones. After following the steps up to
[here](https://kserve.github.io/website/0.13/modelserving/v1beta1/sklearn/v2/#test-the-deployed-model), you should be able to run the following inference request 
(note that we're going to use port-forwarding instead of an ingress port for simplicity)

First check that the sklearn service is running (you have to apply the sklearn.yaml content given in [this section](https://kserve.github.io/website/0.13/modelserving/v1beta1/sklearn/v2/#deploy-the-model-with-rest-endpoint-through-inferenceservice).

```bash
kubectl get inferenceservices.serving.kserve.io sklearn-v2-iris 
NAME              URL                                          READY   PREV   LATEST   PREVROLLEDOUTREVISION   LATESTREADYREVISION               AGE
sklearn-v2-iris   http://sklearn-v2-iris.default.example.com   True           100                             sklearn-v2-iris-predictor-00001   48s
```

Start a k8s port forward

```bash
INGRESS_GATEWAY_SERVICE=$(kubectl get svc --namespace istio-system --selector="app=istio-ingressgateway" --output jsonpath='{.items[0].metadata.name}')
kubectl port-forward --namespace istio-system svc/${INGRESS_GATEWAY_SERVICE} 8080:80
```

In a separate terminal, run an inference (don't forget to create the `iris-input-v2.json`
file using the payload content shown [here](iris-input-v2.json) ):

```bash
curl  \
  -H "Host: ${SERVICE_HOSTNAME}" \
  -H "Content-Type: application/json" \
  -d @./iris-input-v2.json \
  http://localhost:8080/v2/models/sklearn-v2-iris/infer
```

You should get an output that resembles:

```json
{"model_name":"sklearn-v2-iris",
"model_version":null,"id":"1b479ed4-14ca-4b71-8a33-47a7d5c40134",
"parameters":null,
"outputs":[{"name":"output-0",
"shape":[2],
"datatype":"INT32",
"parameters":null,
"data":[1,1]}]}
```

### Deploy the tsfm kserve service

Save the folling yaml snippet to a file called tsfm.yaml:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  annotations:
    serving.kserve.io/deploymentMode: RawDeployment
  name: tsfminferenceserver
spec:
  predictor:
    containers:
      - name: "tsfmgrpcserver"
        image: "localhost:5001/tsfminference:latest"
        imagePullPolicy: Always
        ports:
          - containerPort: 8000
            protocol: TCP
```

Create a namespace for the deployment

```bash
# note that we're using the defualt namespace
# You may not have deploy rights in this namespace on your
# k8s system (you should if you're using kind, though)
kubectl apply -f tsfm.yaml
```

Confirm that the service is running:

```bash
kubectl get inferenceservices.serving.kserve.io tsfminferenceserver 
NAME                  URL                                                  READY   PREV   LATEST   PREVROLLEDOUTREVISION   LATESTREADYREVISION   AGE
tsfminferenceserver   http://tsfminferenceserver-kserve-test.example.com   True                                                                  25m

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
