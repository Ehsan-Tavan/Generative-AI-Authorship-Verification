# docker build -t ghcr.io/pan-webis-de/pan24-generative-authorship-baselines:latest .

FROM nvcr.io/nvidia/cuda:12.3.1-devel-ubuntu22.04

RUN set -x \
    && apt update \
    && apt install -y git python3 python3-packaging python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY . /opt/first_run
WORKDIR /opt/first_run
RUN set -x \
    && pip install --no-cache --upgrade pip \
    && pip install -r requirements.txt

VOLUME /dataset.jsonl
VOLUME /out
