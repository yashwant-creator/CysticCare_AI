#!/bin/bash#!/bin/bash



# Start script for Render deployment# Start script for Render deployment

echo "🚀 Starting CysticCare AI Backend..."echo "🚀 Starting CysticCare AI Backend..."



cd backend/appcd backend/app

uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}

