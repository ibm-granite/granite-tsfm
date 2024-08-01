# based on https://github.com/opendatahub-io/caikit-tgis-serving/blob/main/Dockerfile

FROM registry.access.redhat.com/ubi9/ubi-minimal:latest as builder

RUN microdnf -y update && \
    microdnf -y install \
        git shadow-utils python3.11-pip python-wheel && \
    pip3.11 install --no-cache-dir --upgrade pip wheel && \
    microdnf clean all

ENV POETRY_VIRTUALENVS_IN_PROJECT=1

COPY tsfmservices* /tsfmservices/tsfmservices
COPY pyproject.toml /tsfmservices/
COPY poetry.lock /tsfmservices/
WORKDIR /tsfmservices
# RUN python3.11 -m venv /tsfmservices/.venv
#RUN source /tsfmservices/.venv/bin/activate && \
# pip3.11 install --no-cache-dir ".[services]"
RUN pip3.11 install poetry && poetry install

FROM registry.access.redhat.com/ubi9/ubi-minimal:latest as deploy
RUN microdnf -y update && \
    microdnf -y install \
        shadow-utils python3.11 && \
    microdnf clean all

WORKDIR /tsfmservices

COPY --from=builder /tsfmservices/ /tsfmservices/

ENV VIRTUAL_ENV=/tsfmservices/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV HF_HOME=/tmp

RUN groupadd --system tsfmservices --gid 1001 && \
    adduser --system --uid 1001 --gid 0 --groups tsfmservices \
    --create-home --home-dir /tsfmservices --shell /sbin/nologin \
    --comment "TSFMServices User" tsfmservices

USER tsfmservices

# CMD ["python",  "-m", "tsfmservices.inference.main"]
CMD ["python", "-m", "uvicorn","tsfmservices.inference.main:app", "--host", "0.0.0.0", "--port", "8000" ]