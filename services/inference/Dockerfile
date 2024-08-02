# based on https://github.com/opendatahub-io/caikit-tgis-serving/blob/main/Dockerfile

FROM registry.access.redhat.com/ubi9/ubi-minimal:latest AS builder

RUN microdnf -y update && \
    microdnf -y install \
        git shadow-utils python3.11-pip python-wheel && \
    pip3.11 install --no-cache-dir --upgrade pip wheel && \
    microdnf clean all

ENV POETRY_VIRTUALENVS_IN_PROJECT=1

RUN mkdir /inference
COPY tsfminference/* /inference/tsfminference/
COPY pyproject.toml /inference/
COPY poetry.lock /inference/
WORKDIR /inference
RUN pip3.11 install poetry && poetry install

FROM registry.access.redhat.com/ubi9/ubi-minimal:latest AS deploy
RUN microdnf -y update && \
    microdnf -y install \
        shadow-utils python3.11 && \
    microdnf clean all

WORKDIR /inference

COPY --from=builder /inference /inference

ENV VIRTUAL_ENV=/inference/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV HF_HOME=/tmp

RUN groupadd --system tsfminference --gid 1001 && \
    adduser --system --uid 1001 --gid 0 --groups tsfminference \
    --create-home --home-dir /inference --shell /sbin/nologin \
    --comment "tsfminference User" tsfminference

USER tsfminference

CMD ["python", "-m", "uvicorn","tsfminference.main:app", "--host", "0.0.0.0", "--port", "8000" ]