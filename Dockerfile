# CPU image: API + web UI with the embedding model, cross-encoder and spaCy model baked in.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf TOKENIZERS_PARALLELISM=false
WORKDIR /app

# CPU-only PyTorch keeps the image ~1.5 GB instead of ~6 GB with CUDA wheels.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[models,nlp,llm]" \
 && python -m spacy download en_core_web_sm \
 && python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

ENV HF_HUB_OFFLINE=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["ragmap", "serve", "--host", "0.0.0.0", "--port", "8000", "--preload", "/app/library"]
