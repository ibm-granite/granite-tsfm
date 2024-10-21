# based on https://github.com/opendatahub-io/caikit-tgis-serving/blob/main/Dockerfile

FROM registry.access.redhat.com/ubi9/ubi-minimal:latest
ARG PYTHON_VERSION=3.12

RUN microdnf -y update && \
    microdnf -y install \
        git \
        shadow-utils \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-pip \
        python$(PYTHON_VERSION)-wheel && \
    pip${PYTHON_VERSION} install --no-cache-dir --upgrade pip wheel && \
    microdnf clean all

ENV POETRY_VIRTUALENVS_IN_PROJECT=1

RUN mkdir /finetuning
COPY tsfmfinetuning/* /finetuning/tsfmfinetuning/
COPY pyproject.toml /finetuning/
COPY poetry.lock /finetuning/
WORKDIR /finetuning
RUN pip${PYTHON_VERSION} install poetry && poetry install


ENV VIRTUAL_ENV=/finetuning/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV HF_HOME=/tmp

RUN groupadd --system tsfmfinetuning --gid 1001 && \
    adduser --system --uid 1001 --gid 0 --groups tsfmfinetuning \
    --create-home --home-dir /finetuning --shell /sbin/nologin \
    --comment "tsfmfinetuning User" tsfmfinetuning

USER tsfmfinetuning

# currently finetuning is not a service hosting platform
# CMD ["python","-m","gunicorn","-w","4","-k","uvicorn.workers.UvicornWorker","--bind","0.0.0.0:8000","tsfmfinetuning.main:app"]
