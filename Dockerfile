FROM python:3.11-slim

LABEL org.opencontainers.image.source=https://github.com/jahtz/nlbin
LABEL org.opencontainers.image.description="Generate binary and normalized versions of a set of input images using OCRopus nlbin algorithm."
LABEL org.opencontainers.image.licenses=APACHE-2.0

WORKDIR /build
COPY nlbin ./nlbin
COPY LICENSE .
COPY pyproject.toml .
COPY README.md .
RUN pip install .
RUN nlbin --version

WORKDIR /data
ENTRYPOINT ["nlbin"]
