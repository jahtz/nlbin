FROM python:3.11-slim

WORKDIR /build
COPY nlbin ./nlbin
COPY LICENSE .
COPY pyproject.toml .
COPY README.md .
RUN pip install .

WORKDIR /data
ENTRYPOINT ["nlbin"]
