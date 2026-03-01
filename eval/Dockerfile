FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/workspace/cache \
    HF_HOME=/workspace/cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python deps first (cached unless requirements change)
COPY eval/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy source code
COPY src/ src/
COPY config/ config/
COPY scripts/evaluate.py scripts/evaluate.py
COPY eval/run_benchmarks.py eval/run_benchmarks.py
COPY data/examples/ data/examples/
COPY setup.py ./

RUN pip install --no-cache-dir -e .

ENTRYPOINT ["python", "eval/run_benchmarks.py"]
