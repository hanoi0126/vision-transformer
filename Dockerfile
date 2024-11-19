FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    wget \
    git \
    git-lfs \
    tzdata \
    libffi-dev \
    libssl-dev \
    libbz2-dev \
    zlib1g-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    python-openssl \
    python3-pip \
    python3-setuptools \
    libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# pyenvインストールおよびPythonバージョンの設定
ARG PYENV_ROOT="/root/.pyenv"
ARG PYTHON_VERSION="3.10.11"

RUN git clone --depth=1 https://github.com/pyenv/pyenv.git $PYENV_ROOT

# 環境変数の設定
ENV PATH="$PYENV_ROOT/bin:$PATH"
ENV PYTHON_CONFIGURE_OPTS="--enable-shared"

# Pythonのインストール
RUN eval "$(pyenv init --path)" && \
    pyenv install $PYTHON_VERSION && \
    pyenv global $PYTHON_VERSION && \
    pyenv rehash

# Poetryのインストール
ENV POETRY_HOME="/root/.local" \
    PATH="$POETRY_HOME/bin:$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"

RUN curl -sSL https://install.python-poetry.org | python3 -

# Pythonのパスを追加
ENV PYTHONPATH=/workspace:$PYTHONPATH \
    PATH="/root/.local/bin:$PATH"

WORKDIR /workspace

RUN poetry config virtualenvs.in-project true