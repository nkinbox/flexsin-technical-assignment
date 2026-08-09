# Python 3.12 rather than 3.14: the ecosystem has settled wheels here, and the
# image should not be the place we discover a missing build.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so source edits don't invalidate the install layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-warm the ONNX embedding model (~80 MB) at build time. Without this the
# first upload in a fresh container stalls on a download -- and the container
# would need outbound network access it otherwise never uses.
RUN python -c "from chromadb.utils import embedding_functions; \
embedding_functions.DefaultEmbeddingFunction()(['warmup'])"

COPY app/ ./app/

# Non-root. Two things must be owned by the runtime user:
#
#   /home/appuser/.cache  -- the pre-warmed ONNX model
#   /data                 -- the Chroma mount point. Creating and chowning it
#                            HERE is load-bearing: Docker initialises a fresh
#                            named volume from the image's directory, ownership
#                            included. Without this the volume arrives
#                            root-owned and every upload fails with EACCES.
RUN useradd --create-home --uid 1000 appuser \
    && cp -r /root/.cache /home/appuser/.cache \
    && mkdir -p /data \
    && chown -R appuser:appuser /home/appuser /app /data
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import httpx,sys; \
sys.exit(0 if httpx.get('http://localhost:8000/health', timeout=5).status_code==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
