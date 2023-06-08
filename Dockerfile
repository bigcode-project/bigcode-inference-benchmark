FROM nvcr.io/nvidia/pytorch:23.03-py3

ARG USER=1000
ARG USERNAME=user

WORKDIR /app
ENV PYTHONPATH=/app

RUN useradd -m -u $USER -s /bin/bash $USERNAME \
    && chown $USERNAME /app

# git-lfs is needed to interact with the huggingface hub
# ssl and gcc are needed for text-gen-inference
RUN apt-get update \
    && apt-get install git-lfs libssl-dev gcc \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install


RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && PROTOC_ZIP=protoc-21.12-linux-x86_64.zip \
    && curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP \
    && unzip -o $PROTOC_ZIP -d /usr/local bin/protoc \
    && unzip -o $PROTOC_ZIP -d /usr/local 'include/*' \
    && rm -f $PROTOC_ZIP \
    && chmod 777 /root/ && chmod 777 /root/.cargo

ENV PATH="/root/.cargo/bin:$PATH"

COPY --chown=$USERNAME text-generation-inference/ ./text-generation-inference

RUN cd text-generation-inference && make install && make install-benchmark && cd ..

COPY --chown=$USERNAME ./requirements.txt ./
COPY --chown=$USERNAME transformers/ ./transformers

# Stock version of pip doesn't work with editable transformers.
RUN pip install --upgrade pip --no-cache-dir && pip install -r requirements.txt --no-cache-dir

ENV HUGGINGFACE_HUB_CACHE=/app/data/.hf_cache/

COPY --chown=$USERNAME Makefile .
COPY --chown=$USERNAME src/ ./src
COPY --chown=$USERNAME scripts/ ./scripts
