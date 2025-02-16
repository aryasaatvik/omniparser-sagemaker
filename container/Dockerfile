FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set a docker label to advertise multi-model support on the container
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
# Set a docker label to enable container to use SAGEMAKER_BIND_TO_PORT environment variable if present
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && apt-get clean

# Install system dependencies
RUN apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    python3-pip \
    python3-setuptools \
    gcc \
    wget \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    pkg-config \
    nginx \
    build-essential \
    ca-certificates \
    openjdk-8-jdk-headless \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3.11 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    setuptools \
    wheel \
    mxnet \
    multi-model-server \
    sagemaker-inference \
    retrying

# Set up directories
RUN mkdir -p /opt/ml/model \
    && mkdir -p /home/model-server/tmp

# Install OmniParser dependencies
COPY requirements.txt /home/model-server/requirements.txt
RUN pip install --no-cache-dir -r /home/model-server/requirements.txt

# Copy model server files
COPY container/inference.py /home/model-server/inference.py
COPY container/util /home/model-server/util
COPY container/dockerd-entrypoint.py /home/model-server/dockerd-entrypoint.py

ENTRYPOINT ["python", "/home/model-server/dockerd-entrypoint.py"]

CMD ["serve"]