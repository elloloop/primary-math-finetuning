FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/workspace/cache \
    HF_HOME=/workspace/cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached unless requirements change)
COPY train/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy source code
COPY src/ src/
COPY config/ config/
COPY scripts/train.py scripts/train.py
COPY scripts/quick_test.py scripts/quick_test.py
COPY scripts/evaluate.py scripts/evaluate.py
COPY eval/run_benchmarks.py eval/run_benchmarks.py
COPY data/examples/ data/examples/
COPY runpod_startup.sh setup.py ./

RUN pip install --no-cache-dir -e . && chmod +x runpod_startup.sh

EXPOSE 6006

ENTRYPOINT ["bash", "/app/runpod_startup.sh"]
CMD ["python", "scripts/quick_test.py"]
