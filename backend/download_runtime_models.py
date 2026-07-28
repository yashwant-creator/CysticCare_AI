#!/usr/bin/env python3
"""Download the pinned local reranker during the Docker build."""

import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer


model = os.getenv("RAG_CROSS_ENCODER_MODEL", "ncbi/MedCPT-Cross-Encoder")
revision = os.environ["RAG_CROSS_ENCODER_REVISION"]
AutoTokenizer.from_pretrained(model, revision=revision)
AutoModelForSequenceClassification.from_pretrained(model, revision=revision)
print(f"cached {model}@{revision}")
