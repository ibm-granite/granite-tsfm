# TSFM Finetuning Image

The TSFM Finetuning image provides a runtime for finetuning existing models.
At present we do not support direct service endpoints, the assumption is that
the main `ftmain.py` will serve as the entry point for orchestrated finetuning
workflows initiated from 3rd-party frameworks.
The current implementation includes an example of using the [kubeflow 
training operator (KFTO)](https://github.com/kubeflow/training-operator) and its API for performing
a finetuneing job on a kubernetes-based system.

## Prerequisites:

- GNU make
- git
- git-lfs (available in many system package managers such as apt, dnf, and brew)
- python >=3.10, <3.13
- poetry (`pip install poetry`)
- zsh or bash
- docker or podman (to run examples, we have not tested well with podman)
- kubectl for deploying a local test cluster

## Installation

```sh
pip install poetry && poetry install --with dev
```

### Testing locally

This will run basic unit tests. You should run them and confirm they pass before
proceeding to kubernetes-based tests and examples.

```zsh
make test_local
```

### Building an image

You must have either docker or podman installed on your system for this to
work. You must also have proper permissions on your system to build images. We assume you have a working docker command which can be docker itself
or `podman` that has been aliased as `docker` or has been installed with the podman-docker package that will do this for you.

```zsh
make image
```

Note that be default we build an image **without** GPU support. This makes the development image much smaller
than a fully nvidia-enabled image. GPU enablement is coming soon and will be available via an environment
prefix to the `make image` command.

After a successful build you should have a local image named
`tsfmfinetuning:latest`

```zsh
docker images | grep tsfmfinetuning | head -n 1
tsfmfinetuning                                             latest               df592dcb0533   46 seconds ago      1.49GB
# some of the numeric and hash values on your machine could be different
```

## Running a simple finetuning job on a local kubernetes cluster using kind

For this example we'll use [kind](https://kind.sigs.k8s.io/docs/user/quick-start/),
a lightweight way of running a local kubernetes cluster using docker. We will
use the kubeflow training operator's custom resource to start
and monitor an ayschronous finetuning job.

### Create a local cluster

First:

- [Install kubectl](https://kubernetes.io/docs/tasks/tools/)
- [Install helm](https://helm.sh/docs/intro/install/)
- If you are using podman, you will need to enable the use of an insecure (using http instead of https)
  local container registry by creating a file called `/etc/containers/registries.conf.d/localhost.conf`
  with the following content:

  ```
  [[registry]]
  location = "localhost:5001"
  insecure = true
  ```

- If you're using podman, you may run into issues running the kserve container due to
  open file (nofile) limits. If so,
  see https://github.com/containers/common/blob/main/docs/containers.conf.5.md
  for instructions on how to increase the default limits.

Now install a kind control plane with a local docker registry:

```zsh
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

Confirm that your local cluster is running:

```zsh
kubectl config use-context kind-kind
Switched to context "kind-kind".
```

```zsh
kubectl get pods -A
NAMESPACE            NAME                                         READY   STATUS    RESTARTS   AGE
kube-system          coredns-7c65d6cfc9-b8fcf                     1/1     Running   0          16h
kube-system          coredns-7c65d6cfc9-bptc4                     1/1     Running   0          16h
kube-system          etcd-kind-control-plane                      1/1     Running   0          16h
kube-system          kindnet-7xz78                                1/1     Running   0          16h
kube-system          kube-apiserver-kind-control-plane            1/1     Running   0          16h
kube-system          kube-controller-manager-kind-control-plane   1/1     Running   0          16h
kube-system          kube-proxy-nwwvq                             1/1     Running   0          16h
kube-system          kube-scheduler-kind-control-plane            1/1     Running   0          16h
local-path-storage   local-path-provisioner-57c5987fd4-ts26j      1/1     Running   0          16h

```

Note that your names will look similar to necessarily identical to the above.

### Set up rancher storage provisioning (this is necessary only when using a kind local cluster)

```zsh
kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/v0.0.30/deploy/local-path-storage.yaml
```

### Install the kubeflow training operator (KFTO)

The following steps are based on those presented in [the official KFTO documentation](https://www.kubeflow.org/docs/components/training/installation/).

```zsh
kubectl apply --server-side -k "github.com/kubeflow/training-operator.git/manifests/overlays/standalone?ref=v1.8.1"
```

Check that the training operator pod is running:

```zsh
kubectl get pods -n kubeflow

kubeflow             training-operator-7f8bfd56f-lrpm2            1/1     Running   0          3m41s
```

As before, your output should be similar but not necessarily identical to the above.

Check that the custom resource definitions have been created:

```zsh
kubectl get crd

NAME                       CREATED AT
mpijobs.kubeflow.org       2024-12-07T18:21:04Z
mxjobs.kubeflow.org        2024-12-07T18:21:05Z
paddlejobs.kubeflow.org    2024-12-07T18:21:05Z
pytorchjobs.kubeflow.org   2024-12-07T18:21:06Z
tfjobs.kubeflow.org        2024-12-07T18:21:06Z
xgboostjobs.kubeflow.org   2024-12-07T18:21:06Z

```

### Push the tsfm service image to the kind local registry:

```zsh
# don't forget to run "make image" first
docker tag tsfmfinetuning:latest localhost:5001/tsfmfinetuning:latest
  docker push localhost:5001/tsfmfinetuning:latest
```

### Create your local storage

Define a persistent volume claim using rancher's local-path storage:

```sh
kubectl apply -f examples/local_pvc.yaml
```

Create a alpine instance bound to this PVC to make it easier to copy things to the local storage location

```sh
kubectl apply -f examples/alpine.yaml
```

Copy models to the PVC

```sh
make clone_models && make fetchdata
```

Copy data and payload parameters to the storage location. **Remember this is
just a local development example, you would not be doing things like
giving rwX permission to everyone (the last line in the code snippet below)
in a real deployment!**

```sh
kubectl cp mytest-tsfm alpine:/data \
&& kubectl cp --no-preserve=true data/ETTh1.csv alpine:/data \
&& tf=$(mktemp) \
&& cat data/ftpayload.json | awk '{gsub("file://./", "file:///")}1' >> $tf \
&& kubectl cp --no-preserve=true $tf alpine:/data/ftpayload.json \
&& cat tsfmfinetuning/default_config.yml | awk '{gsub("/tmp", "/data")}1' > $tf \
&& kubectl cp --no-preserve=true $tf alpine:/data/default_config.yml \
&& kubectl exec alpine -- chmod -R go+rwX /data
```

Create a finetuning job and monitor its output

```sh
kubectl apply -f examples/kfto_job.yaml

pytorchjob.kubeflow.org/tsfmfinetuning-job created
```

```sh
kubectl logs -f tsfmfinetuning-job-master-0

/finetuning/.venv/lib/python3.12/site-packages/pydantic/_internal/_fields.py:192: UserWarning: Field name "schema" in "ForecastingInferenceInput" shadows an attribute in parent "BaseInferenceInput"
  warnings.warn(
/finetuning/.venv/lib/python3.12/site-packages/pydantic/_internal/_fields.py:192: UserWarning: Field name "schema" in "ForecastingTuneInput" shadows an attribute in parent "BaseTuneInput"
  warnings.warn(
INFO:p-1:t-139742116783936:finetuning.py:__init__:registered tinytimemixer
INFO:p-1:t-139742116783936:finetuning.py:_finetuning_common:in _forecasting_tuning_workflow
INFO:p-1:t-139742116783936:finetuning.py:load:No preprocessor found
INFO:p-1:t-139742116783936:hfutil.py:load_model:Found model class: TinyTimeMixerForPrediction
INFO:p-1:t-139742116783936:finetuning.py:load:Successfully loaded model
WARNING:p-1:t-139742116783936:other.py:check_os_kernel:Detected kernel version 4.18.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.
INFO:p-1:t-139742116783936:finetuning.py:_finetuning_common:calling trainer.train
{'loss': 7.3424, 'grad_norm': 8.357532501220703, 'learning_rate': 0.0, 'epoch': 1.0}

100%|██████████| 3/3 [00:17<00:00,  3.51s/it]e': 6.2719, 'eval_samples_per_second': 119.581, 'eval_steps_per_second': 3.827, 'epoch': 1.0}
100%|██████████| 3/3 [00:17<00:00,  5.85s/it]_second': 42.739, 'train_steps_per_second': 0.171, 'train_loss': 7.342405319213867, 'epoch': 1.0}
INFO:p-1:t-139742116783936:finetuning.py:_finetuning_common:done with training
```

Confirm that a new finetuned model has been produced

```sh
# 'finetuned_from_kfto' comes from the value set for the
# --model_name argument in examples/kfto_job.yaml
kubectl exec alpine -- ls -lR /data/finetuned_from_kfto

/data/finetuned_from_kfto:
total 3188
-rw-r--r--    1 1001     root          1573 Dec  9 15:28 config.json
-rw-r--r--    1 1001     root            69 Dec  9 15:28 generation_config.json
-rw-r--r--    1 1001     root       3240592 Dec  9 15:28 model.safetensors
-rw-r--r--    1 1001     root           857 Dec  9 15:28 preprocessor_config.json
-rw-r--r--    1 1001     root          5304 Dec  9 15:28 training_args.bin

```
