FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.11

LABEL org.opencontainers.image.source=https://github.com/jahtz/nlbin
LABEL org.opencontainers.image.description="Generate binary and normalized versions of a set of input images using OCRopus nlbin algorithm."
LABEL org.opencontainers.image.licenses=APACHE-2.0

RUN apt-get update && apt-get install -y software-properties-common curl && \
    add-apt-repository ppa:deadsnakes/ppa -y && apt-get update && \
    apt-get install -y build-essential python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python3-pip && \
    curl -O https://bootstrap.pypa.io/get-pip.py && python${PYTHON_VERSION} get-pip.py && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY nlbin ./nlbin
COPY LICENSE .
COPY pyproject.toml .
COPY README.md .
RUN pip${PYTHON_VERSION} install .[cuda12]

WORKDIR /data
ENTRYPOINT ["nlbin"]
