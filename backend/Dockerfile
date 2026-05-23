# Use Python 3.11 slim image
FROM python:3.11-slim

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY app/requirements_openai.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements_openai.txt

# Copy the entire app directory
COPY app/ ./app/

# Expose port 8080 (Cloud Run default)
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application as a package so relative imports resolve correctly
CMD ["uvicorn", "app.main_openai:app", "--host", "0.0.0.0", "--port", "8080"]
