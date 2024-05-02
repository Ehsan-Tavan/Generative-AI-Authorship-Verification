# docker build -t pan24-generative-authorship .

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ADD requirements.txt /

RUN set -x \
    && apt update \
    && apt install -y git python3 python3-packaging python3-pip \
    && pip install --no-cache --upgrade pip \
    && pip install -r /requirements.txt \
    && rm -Rr /requirements.txt \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/app

COPY . /app

