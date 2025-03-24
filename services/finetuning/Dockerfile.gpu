# Copyright contributors to the TSFM project

# Portions of this Dockerfile are taken from
# https://gitlab.com/nvidia/container-images
# Copyright (c) 2019,2020,2021 NVIDIA CORPORATION. All rights reserved.

FROM registry.access.redhat.com/ubi9/ubi:latest AS base

ENV NVARCH x86_64
ENV NVIDIA_REQUIRE_CUDA "cuda>=12.4 brand=tesla,driver>=470,driver<471 brand=unknown,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=geforce,driver>=470,driver<471 brand=geforcertx,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=titan,driver>=470,driver<471 brand=titanrtx,driver>=470,driver<471 brand=tesla,driver>=525,driver<526 brand=unknown,driver>=525,driver<526 brand=nvidia,driver>=525,driver<526 brand=nvidiartx,driver>=525,driver<526 brand=geforce,driver>=525,driver<526 brand=geforcertx,driver>=525,driver<526 brand=quadro,driver>=525,driver<526 brand=quadrortx,driver>=525,driver<526 brand=titan,driver>=525,driver<526 brand=titanrtx,driver>=525,driver<526 brand=tesla,driver>=535,driver<536 brand=unknown,driver>=535,driver<536 brand=nvidia,driver>=535,driver<536 brand=nvidiartx,driver>=535,driver<536 brand=geforce,driver>=535,driver<536 brand=geforcertx,driver>=535,driver<536 brand=quadro,driver>=535,driver<536 brand=quadrortx,driver>=535,driver<536 brand=titan,driver>=535,driver<536 brand=titanrtx,driver>=535,driver<536"
ENV NV_CUDA_CUDART_VERSION 12.4.127-1
COPY cuda.repo-x86_64 /etc/yum.repos.d/cuda.repo
RUN NVIDIA_GPGKEY_SUM=d0664fbbdb8c32356d45de36c5984617217b2d0bef41b93ccecd326ba3b80c87 && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/rhel9/${NVARCH}/D42D0685.pub | sed '/^Version/d' > /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA && \
    echo "$NVIDIA_GPGKEY_SUM  /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA" | sha256sum -c --strict -
ENV CUDA_VERSION 12.4.1

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN yum upgrade -y && yum install -y \
    cuda-cudart-12-4-${NV_CUDA_CUDART_VERSION} \
    cuda-compat-12-4 \
    && dnf remove -y --disableplugin=subscription-manager \
    subscription-manager \
    && yum clean all \
    && dnf clean all \
    && rm -rf /var/cache/yum/*

# nvidia-docker 1.0
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

FROM base AS runtime-base

ENV NV_CUDA_LIB_VERSION 12.4.1-1
ENV NV_NVTX_VERSION 12.4.127-1
ENV NV_LIBNPP_VERSION 12.2.5.30-1
ENV NV_LIBNPP_PACKAGE libnpp-12-4-${NV_LIBNPP_VERSION}
ENV NV_LIBCUBLAS_VERSION 12.4.5.8-1
ENV NV_LIBNCCL_PACKAGE_NAME libnccl
ENV NV_LIBNCCL_PACKAGE_VERSION 2.21.5-1
ENV NV_LIBNCCL_VERSION 2.21.5
ENV NCCL_VERSION 2.21.5
ENV NV_LIBNCCL_PACKAGE ${NV_LIBNCCL_PACKAGE_NAME}-${NV_LIBNCCL_PACKAGE_VERSION}+cuda12.4

RUN yum install -y \
    cuda-libraries-12-4-${NV_CUDA_LIB_VERSION} \
    cuda-nvtx-12-4-${NV_NVTX_VERSION} \
    ${NV_LIBNPP_PACKAGE} \
    libcublas-12-4-${NV_LIBCUBLAS_VERSION} \
    ${NV_LIBNCCL_PACKAGE} \
    && yum clean all \
    && rm -rf /var/cache/yum/*

ENV NV_CUDNN_VERSION 9.1.0.70-1
ENV NV_CUDNN_PACKAGE libcudnn9-cuda-12-${NV_CUDNN_VERSION}
LABEL com.nvidia.cudnn.version="${NV_CUDNN_VERSION}"

RUN yum install -y \
    ${NV_CUDNN_PACKAGE} \
    && yum clean all \
    && rm -rf /var/cache/yum/*

FROM runtime-base AS python-install
ARG PYTHON_VERSION=3.12

ARG CODEDIR
RUN yum install -y python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-pip \
    python${PYTHON_VERSION}-wheel \
    && yum clean all \
    && ln -s /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python \
    && ln -s /usr/bin/pip-${PYTHON_VERSION} /usr/local/bin/pip \
    && groupadd --system ${CODEDIR} --gid 1001 \
    && rm -f /usr/share/doc/perl-Net-SSLeay/examples/server_key.pem \
    && adduser --system --uid 1001 --gid 0 --groups ${CODEDIR} \
    --create-home --home-dir /${CODEDIR} --shell /sbin/nologin \
    --comment "${CODEDIR} User" ${CODEDIR}

# ^^
# perl-Net-SSLeay.x86_64 and server_key.pem are installed with git as dependencies
# Twistlock detects it as H severity: Private keys stored in image

FROM python-install AS tsfm-install
ENV LD_LIBRARY_PATH=/usr/local/lib/python${PYTHON_VERSION}/site-packages/nvidia/cuda_cupti/lib:/tsfm/.venv/lib/python${PYTHON_VERSION}/site-packages/nvidia/cuda_cupti/lib:/tsfm/.venv/lib/python${PYTHON_VERSION}/site-packages/cusparselt/lib/:${LD_LIBRARY_PATH}

RUN yum install -y git shadow-utils \
    && yum clean all

ENV POETRY_VIRTUALENVS_IN_PROJECT=1

RUN mkdir /tsfm
COPY ${CODEDIR}/* /tsfm/${CODEDIR}/
COPY pyproject.toml /tsfm/
COPY poetry.lock /tsfm/
WORKDIR /tsfm
RUN pip install --no-cache-dir poetry && poetry install --no-cache \
    && rm -rf /root/.cache || true \
    && rm -rf /${CODEDIR}/.cache || true \
    && yum remove -y git shadow-utils && yum clean all \
    && chown -R ${CODEDIR} /tsfm \
    && rpm -e $(dnf repoquery python3-* -q --installed) dnf python3 yum crypto-policies-scripts

ENV VIRTUAL_ENV=/tsfm/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV HF_HOME=/tmp

USER ${CODEDIR}
