# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY app/requirements_openai.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements_openai.txt

# Copy the entire app directory
COPY app/ ./app/

# Copy the pre-built ChromaDB vector database
# This avoids reprocessing PDFs on every container startup
COPY app/openai_chroma_data/ ./app/openai_chroma_data/

# Copy .env file (optional - better to use Cloud Run env vars in production)
# Comment out if you don't want to include .env in the container
COPY .env* ./

# Expose port 8080 (Cloud Run default)
EXPOSE 8080

# Change to the app directory for proper imports
WORKDIR /app/app

# Run the application
# Using main_openai:app since we're now in the app directory
CMD ["uvicorn", "main_openai:app", "--host", "0.0.0.0", "--port", "8080"]
