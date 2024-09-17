# based on https://github.com/opendatahub-io/caikit-tgis-serving/blob/main/Dockerfile

FROM registry.access.redhat.com/ubi9/ubi-minimal:latest AS builder

RUN microdnf -y update && \
    microdnf -y install \
        git shadow-utils python3.11-pip python-wheel && \
    pip3.11 install --no-cache-dir --upgrade pip wheel && \
    microdnf clean all

ENV POETRY_VIRTUALENVS_IN_PROJECT=1

RUN mkdir /finetuning
COPY tsfmfinetuning/* /finetuning/tsfmfinetuning/
COPY pyproject.toml /finetuning/
COPY poetry.lock /finetuning/
WORKDIR /finetuning
RUN pip3.11 install poetry && poetry install

FROM registry.access.redhat.com/ubi9/ubi-minimal:latest AS deploy
RUN microdnf -y update && \
    microdnf -y install \
        shadow-utils python3.11 && \
    microdnf clean all

WORKDIR /finetuning

COPY --from=builder /finetuning /finetuning

ENV VIRTUAL_ENV=/finetuning/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV HF_HOME=/tmp

RUN groupadd --system tsfmfinetuning --gid 1001 && \
    adduser --system --uid 1001 --gid 0 --groups tsfmfinetuning \
    --create-home --home-dir /finetuning --shell /sbin/nologin \
    --comment "tsfmfinetuning User" tsfmfinetuning

USER tsfmfinetuning

CMD ["python", "-m", "uvicorn","tsfmfinetuning.main:app", "--host", "0.0.0.0", "--port", "8000" ]