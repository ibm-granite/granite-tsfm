FROM registry.access.redhat.com/ubi9/ubi:latest AS base

ARG PYTHON_VERSION=3.12
RUN dnf remove -y --disableplugin=subscription-manager \
    subscription-manager \
    &&\
    dnf install -y \
    python${PYTHON_VERSION} \
    procps \
    g++ \
    python${PYTHON_VERSION}-devel \
    python$(PYTHON_VERSION)-wheel \
    python${PYTHON_VERSION}-pip \
    && python${PYTHON_VERSION} -m ensurepip --upgrade \
    && python${PYTHON_VERSION} -m pip install --upgrade pip wheel \
    && python${PYTHON_VERSION} -m pip install --upgrade setuptools \
    && dnf update -y \
    && dnf clean all

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8


RUN useradd -u 1001 tsfmfinetuning -m -g 0 --system && \
    chmod g+rx /home/tsfmfinetuning

FROM base AS release-base

# Removes the python3.9 code to eliminate possible CVEs.  Also removes dnf
RUN rpm -e $(dnf repoquery python3-* -q --installed) dnf python3 yum crypto-policies-scripts

## CUDA Base ###################################################################
FROM base AS cuda-base

# Ref: https://docs.nvidia.com/cuda/archive/12.1.0/cuda-toolkit-release-notes/
ENV CUDA_VERSION=12.1.0 \
    NV_CUDA_LIB_VERSION=12.1.0-1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NV_CUDA_CUDART_VERSION=12.1.55-1 \
    NV_CUDA_COMPAT_VERSION=530.30.02-1

RUN dnf config-manager \
    --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo \
    && dnf install -y \
    cuda-cudart-12-1-${NV_CUDA_CUDART_VERSION} \
    cuda-compat-12-1-${NV_CUDA_COMPAT_VERSION} \
    && echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf \
    && dnf clean all

ENV CUDA_HOME="/usr/local/cuda" \
    PATH="/usr/local/nvidia/bin:${CUDA_HOME}/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"


## CUDA Development ############################################################
FROM cuda-base AS cuda-devel

# Ref: https://developer.nvidia.com/nccl/nccl-legacy-downloads
ENV NV_CUDA_CUDART_DEV_VERSION=12.1.55-1 \
    NV_NVML_DEV_VERSION=12.1.55-1 \
    NV_LIBCUBLAS_DEV_VERSION=12.1.0.26-1 \
    NV_LIBNPP_DEV_VERSION=12.0.2.50-1 \
    NV_LIBNCCL_DEV_PACKAGE_VERSION=2.18.3-1+cuda12.1

RUN dnf config-manager \
    --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo \
    && dnf install -y \
    cuda-command-line-tools-12-1-${NV_CUDA_LIB_VERSION} \
    cuda-libraries-devel-12-1-${NV_CUDA_LIB_VERSION} \
    cuda-minimal-build-12-1-${NV_CUDA_LIB_VERSION} \
    cuda-cudart-devel-12-1-${NV_CUDA_CUDART_DEV_VERSION} \
    cuda-nvml-devel-12-1-${NV_NVML_DEV_VERSION} \
    libcublas-devel-12-1-${NV_LIBCUBLAS_DEV_VERSION} \
    libnpp-devel-12-1-${NV_LIBNPP_DEV_VERSION} \
    libnccl-devel-${NV_LIBNCCL_DEV_PACKAGE_VERSION} \
    && dnf clean all

ENV LIBRARY_PATH="$CUDA_HOME/lib64/stubs"

FROM cuda-devel AS python-installations

RUN dnf install -y git && \
    # perl-Net-SSLeay.x86_64 and server_key.pem are installed with git as dependencies
    # Twistlock detects it as H severity: Private keys stored in image
    rm -f /usr/share/doc/perl-Net-SSLeay/examples/server_key.pem && \
    dnf clean all

USER tsfmfinetuning
# ##############TSFM INSTALL
ENV POETRY_VIRTUALENVS_IN_PROJECT=1
ENV PATH=${PATH}:/home/tsfmfinetuning/.local/bin
ENV VIRTUAL_ENV=/home/tsfmfinetuning/.venv
ENV PATH=${VIRTUAL_ENV}/bin:$PATH
ENV HF_HOME=/tmp

COPY tsfmfinetuning/* /home/tsfmfinetuning/tsfmfinetuning/
COPY pyproject.toml /home/tsfmfinetuning/
COPY poetry.lock /home/tsfmfinetuning/
WORKDIR /home/tsfmfinetuning
RUN pip${PYTHON_VERSION} install poetry && poetry install


#RUN groupadd --system tsfmfinetuning --gid 1001 && \
#    adduser --system --uid 1001 --gid 0 --groups tsfmfinetuning \
#    --create-home --home-dir /finetuning --shell /sbin/nologin \
#    --comment "tsfmfinetuning User" tsfmfinetuning

# USER tsfmfinetuning

# currently finetuning is not a service hosting platform
# CMD ["python","-m","gunicorn","-w","4","-k","uvicorn.workers.UvicornWorker","--bind","0.0.0.0:8000","tsfmfinetuning.main:app"]
