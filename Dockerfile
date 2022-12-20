FROM nvcr.io/nvidia/pytorch:22.11-py3

WORKDIR /app
ENV PYTHONPATH=/app

COPY ./requirements.txt ./
COPY transformers/ ./transformers
RUN pip install -r requirements.txt

COPY src/ ./src
