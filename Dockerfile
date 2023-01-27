FROM nvcr.io/nvidia/pytorch:22.11-py3

ARG USER=1000
ARG USERNAME=user

WORKDIR /app
ENV PYTHONPATH=/app

RUN useradd -m -u $USER -s /bin/bash $USERNAME \
    && chown $USERNAME /app

COPY --chown=$USERNAME ./requirements.txt ./
COPY --chown=$USERNAME transformers/ ./transformers

# Stock version of pip doesn't work with editable transformers.
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY --chown=$USERNAME src/ ./src
