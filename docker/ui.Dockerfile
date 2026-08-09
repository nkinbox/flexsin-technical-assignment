FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# The UI is a thin HTTP client -- it needs neither the vector store nor the
# model SDK, so it installs a narrow dependency set rather than requirements.txt.
RUN pip install --no-cache-dir streamlit==1.61.1 httpx==0.28.1

COPY ui/ ./ui/

RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import httpx,sys; \
sys.exit(0 if httpx.get('http://localhost:8501/_stcore/health', timeout=5).status_code==200 else 1)"

CMD ["streamlit", "run", "ui/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
