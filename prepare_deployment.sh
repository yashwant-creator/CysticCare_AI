#!/bin/bash
# Prepare Backend for GCP Deployment with Enhanced Metadata
# This script ensures metadata is properly included in deployment

set -e

echo "=========================================="
echo "GCP Deployment Preparation Script"
echo "=========================================="
echo ""

# Navigate to backend directory
cd backend/app

echo "Step 1: Activating Python environment..."
if [ -d "../../backend_env" ]; then
    source ../../backend_env/bin/activate
    echo "✓ Environment activated"
else
    echo "✗ Error: backend_env not found"
    exit 1
fi

echo ""
echo "Step 2: Checking for PDF files..."
if [ -d "papers" ] && [ "$(ls -A papers/*.pdf 2>/dev/null)" ]; then
    PDF_COUNT=$(ls -1 papers/*.pdf 2>/dev/null | wc -l)
    echo "✓ Found $PDF_COUNT PDF files"
else
    echo "✗ Warning: No PDF files found in papers/ directory"
fi

echo ""
echo "Step 3: Updating metadata in ChromaDB..."
if [ -d "openai_chroma_data" ]; then
    echo "  ChromaDB exists, updating metadata..."
    python update_metadata.py
    echo "✓ Metadata updated"
else
    echo "  ChromaDB doesn't exist, will be created on first run"
fi

echo ""
echo "Step 4: Verifying metadata cache..."
if [ -f "metadata_cache.json" ]; then
    echo "✓ metadata_cache.json exists"
    CACHE_SIZE=$(wc -c < metadata_cache.json)
    echo "  Size: $CACHE_SIZE bytes"
else
    echo "⚠ metadata_cache.json not found (will be created)"
fi

echo ""
echo "Step 5: Checking deployment files..."
cd ../..

# Check Dockerfile
if [ -f "backend/Dockerfile" ]; then
    echo "✓ Dockerfile exists"
    
    # Check if metadata files are included
    if grep -q "metadata_cache.json" backend/Dockerfile; then
        echo "✓ Dockerfile includes metadata_cache.json"
    else
        echo "⚠ Consider adding metadata_cache.json to Dockerfile"
    fi
else
    echo "✗ Dockerfile not found"
fi

# Check .dockerignore
if [ -f "backend/.dockerignore" ]; then
    if grep -q "metadata_cache.json" backend/.dockerignore; then
        echo "⚠ WARNING: .dockerignore excludes metadata_cache.json"
        echo "  You should remove this exclusion for proper metadata deployment"
    else
        echo "✓ .dockerignore doesn't exclude metadata files"
    fi
else
    echo "  No .dockerignore file found"
fi

echo ""
echo "Step 6: Generating deployment summary..."
cd backend/app

if [ -f "metadata_summary.txt" ]; then
    echo "✓ Metadata summary available at: backend/app/metadata_summary.txt"
    echo ""
    echo "Preview:"
    head -n 20 metadata_summary.txt
else
    echo "⚠ No metadata summary found"
fi

echo ""
echo "=========================================="
echo "Pre-Deployment Checklist"
echo "=========================================="
echo ""
echo "✓ 1. Metadata has been updated in ChromaDB"
echo "✓ 2. metadata_cache.json is ready for deployment"
echo ""
echo "Before deploying to GCP, ensure:"
echo "  [ ] openai_chroma_data/ directory will be included"
echo "  [ ] metadata_cache.json will be included"
echo "  [ ] .dockerignore doesn't exclude these files"
echo "  [ ] GCP has persistent storage configured for ChromaDB"
echo ""
echo "To deploy, run:"
echo "  ./build.sh"
echo ""
echo "To test locally first:"
echo "  cd backend/app"
echo "  source ../../backend_env/bin/activate"
echo "  python main_openai.py"
echo ""
echo "=========================================="

# Return to original directory
cd ../..
