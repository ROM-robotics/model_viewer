# Ubuntu 22.04 base, no venv: all Python deps are installed straight into the
# system interpreter (python3.10, Ubuntu 22.04's default).
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (better layer caching on source changes). torch's CPU-only
# wheel lives on a dedicated index; PyPI is kept as a fallback for everything else.
COPY requirements.txt .
RUN pip3 install \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r requirements.txt

COPY . .

RUN mkdir -p static/uploads

EXPOSE 5000

# Run with `--network host` (see docker/README.md): Netron's detailed-view
# server binds a random localhost port per upload, so the container must share
# the host's network namespace instead of relying on `-p` port publishing.
CMD ["python3", "app.py"]
