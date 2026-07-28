# Use Python 3.11 slim image
FROM python:3.11-slim

ARG MEDCPT_REVISION=71caf65d4927987813984f54c284405a13fcca49
ENV RAG_CROSS_ENCODER_MODEL=ncbi/MedCPT-Cross-Encoder \
    RAG_CROSS_ENCODER_REVISION=${MEDCPT_REVISION} \
    RAG_CROSS_ENCODER_LOCAL_ONLY=true \
    RAG_CROSS_ENCODER_FALLBACK_TO_LLM=false \
    RAG_RERANKER_BACKEND=medcpt \
    RAG_HYDE_ENABLED=false \
    PRELOAD_MEDCPT=true \
    HELPER_CHAT_MODEL=gpt-4o-mini \
    ANSWER_REASONING_EFFORT=none \
    ANSWER_MAX_TOKENS=700 \
    ANSWER_VERBOSITY=low \
    HELPER_MAX_TOKENS=350 \
    PLANNER_MAX_TOKENS=180 \
    RAG_RERANK_CANDIDATES=8 \
    RAG_CROSS_ENCODER_MAX_LENGTH=192 \
    RAG_CROSS_ENCODER_BATCH_SIZE=8 \
    DAILY_OPENAI_CALL_LIMIT=10000 \
    DAILY_OPENAI_TOKEN_LIMIT=5000000 \
    TRACEAI_ENABLED=false \
    TRACEAI_CAPTURE_CONTENT=false \
    HF_HOME=/opt/huggingface

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY app/requirements_openai.txt .

# Install dependencies
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1
RUN pip install --no-cache-dir -r requirements_openai.txt

# Cache the pinned reranker at build time. Runtime model loading is offline.
COPY download_runtime_models.py .
RUN test -n "$MEDCPT_REVISION" && python download_runtime_models.py

# Copy application code and the prebuilt Chroma database into the image.
COPY app/ ./app/
COPY verify_baked_index.py .

# The image is invalid if the baked index is missing, partial, or mismatched.
RUN python verify_baked_index.py
RUN chmod -R a-w app/openai_chroma_data app/index_manifest.json

# Expose port 8080 (Cloud Run default)
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/live || exit 1

# Run the application as a package so relative imports resolve correctly
CMD ["uvicorn", "app.main_openai:app", "--host", "0.0.0.0", "--port", "8080"]
