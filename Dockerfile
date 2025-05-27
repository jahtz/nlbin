ARG IMAGE="python:3.11-slim"

FROM ${IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

LABEL org.opencontainers.image.title="nlbin"
LABEL org.opencontainers.image.source=https://github.com/jahtz/nlbin
LABEL org.opencontainers.image.description="Generate binary and normalized versions of a set of input images using OCRopus nlbin algorithm."
LABEL org.opencontainers.image.licenses=APACHE-2.0

WORKDIR /build
COPY cli ./cli
COPY nlbin ./nlbin
COPY LICENSE .
COPY pyproject.toml .
COPY README.md .
RUN pip install .

RUN nlbin --version

WORKDIR /data
ENTRYPOINT [ "nlbin" ]
