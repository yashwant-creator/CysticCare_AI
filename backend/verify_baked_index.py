#!/usr/bin/env python3
"""Validate the baked vector database without using OpenAI or modifying it."""

from app.services.index_manifest import verify_baked_index


if __name__ == "__main__":
    verified = verify_baked_index(verify_checksum=True)
    manifest = verified["manifest"]
    print(
        f"verified {manifest['collection_name']} "
        f"schema={manifest['index_schema_version']} "
        f"vectors={manifest['vector_count']}"
    )
