ARG IMAGE="nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04"

FROM ${IMAGE}

ARG PIPARG="cuda12"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION="3.11"
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

LABEL org.opencontainers.image.title="nlbin"
LABEL org.opencontainers.image.source=https://github.com/jahtz/nlbin
LABEL org.opencontainers.image.description="Generate binary and normalized versions of a set of input images using OCRopus nlbin algorithm."
LABEL org.opencontainers.image.licenses=APACHE-2.0

RUN apt-get update && apt-get install -y software-properties-common curl git && \
    add-apt-repository ppa:deadsnakes/ppa -y && apt-get update && \
    apt-get install -y build-essential python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python3-pip && \
    curl -O https://bootstrap.pypa.io/get-pip.py && python${PYTHON_VERSION} get-pip.py && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY cli ./cli
COPY nlbin ./nlbin
COPY LICENSE .
COPY pyproject.toml .
COPY README.md .
RUN pip${PYTHON_VERSION} install .[${PIPARG}]

RUN nlbin --version

WORKDIR /data
ENTRYPOINT [ "nlbin" ]
