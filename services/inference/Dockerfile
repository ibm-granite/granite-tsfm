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
    microdnf remove python3.9 python-unversioned-command python3-setuptools python3-wheel -y && \
    microdnf clean all

ENV POETRY_VIRTUALENVS_IN_PROJECT=1

RUN mkdir /inference
COPY tsfminference/* /inference/tsfminference/
COPY pyproject.toml /inference/
COPY poetry.lock /inference/
WORKDIR /inference
RUN pip${PYTHON_VERSION} install "poetry>=2.0.0" && poetry install


ENV VIRTUAL_ENV=/inference/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV HF_HOME=/tmp

RUN groupadd --system tsfminference --gid 1001 && \
    adduser --system --uid 1001 --gid 0 --groups tsfminference \
    --create-home --home-dir /inference --shell /sbin/nologin \
    --comment "tsfminference User" tsfminference

USER tsfminference

CMD ["python","-m","gunicorn","-w","4","-k","uvicorn.workers.UvicornWorker","--bind","0.0.0.0:8000","tsfminference.main:app"]
