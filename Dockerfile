FROM nvcr.io/nvidia/pytorch:23.03-py3

ARG USER=1000
ARG USERNAME=user

WORKDIR /app
ENV PYTHONPATH=/app

RUN useradd -m -u $USER -s /bin/bash $USERNAME \
    && chown $USERNAME /app

# git-lfs is needed to interact with the huggingface hub
RUN apt-get update \
    && apt-get install git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

COPY --chown=$USERNAME ./requirements.txt ./
COPY --chown=$USERNAME transformers/ ./transformers

# Stock version of pip doesn't work with editable transformers.
RUN pip install --upgrade pip --no-cache-dir && pip install -r requirements.txt --no-cache-dir

COPY --chown=$USERNAME Makefile .
COPY --chown=$USERNAME src/ ./src
COPY --chown=$USERNAME scripts/ ./scripts
