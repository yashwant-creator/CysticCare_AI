"""
Manual test harness for the Vision LLM feature.

Exercises the real production code path used during RAG ingestion:
    extract_images_from_pdf()  ->  OpenAIService.describe_image()

This makes real (billable) OpenAI Vision API calls using OPENAI_API_KEY
from backend/.env. Run from the `backend/` directory:

    .venv/bin/python test_vision_feature.py [path/to/paper.pdf] [--max N]

Defaults: a sample PKD paper, up to 3 images described.
"""

import argparse
import os
import sys
import time

from dotenv import load_dotenv

# Load backend/.env (same as main_openai.py)
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from app.services.openai_service import OpenAIService
from app.utils.openai_utils import (
    extract_images_from_pdf,
    load_session_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test the Vision LLM feature.")
    parser.add_argument(
        "pdf",
        nargs="?",
        default=os.path.join("app", "papers", "Booij_2019.pdf"),
        help="Path to a PDF to extract figures from.",
    )
    parser.add_argument(
        "--max", type=int, default=3, help="Max number of images to describe."
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set (checked backend/.env).")
        return 1
    if not os.path.exists(args.pdf):
        print(f"ERROR: PDF not found: {args.pdf}")
        return 1

    config = load_session_config()
    print("=" * 70)
    print("VISION LLM FEATURE TEST")
    print("=" * 70)
    print(f"PDF                : {args.pdf}")
    print(f"vision_model       : {config['vision_model']}")
    print(f"image descriptions : {config['enable_image_descriptions']}")
    print("-" * 70)

    service = OpenAIService(
        embedding_model=config["embedding_model"],
        chat_model=config["chat_model"],
        vision_model=config["vision_model"],
        max_retries=config["max_retries"],
        retry_delay=config["retry_delay"],
    )

    print("Extracting candidate images from PDF...")
    images = extract_images_from_pdf(args.pdf)
    print(f"Found {len(images)} candidate image(s) after size/aspect filtering.\n")
    if not images:
        print("No images to describe. Try a PDF with figures.")
        return 0

    described = 0
    non_informative = 0
    failures = 0

    for idx, image in enumerate(images[: args.max], start=1):
        page = image["page_number"]
        caption = (image["caption"] or "").strip()
        size_kb = len(image["image_bytes"]) / 1024
        print(f"[{idx}] page {page} | {image['ext']} | {size_kb:.0f} KB")
        if caption:
            preview = caption if len(caption) <= 120 else caption[:117] + "..."
            print(f'    caption hint: "{preview}"')

        t0 = time.time()
        try:
            description = service.describe_image(
                image_bytes=image["image_bytes"],
                image_ext=image["ext"],
                caption=caption,
            )
        except Exception as e:
            failures += 1
            print(f"    -> ERROR after {time.time() - t0:.1f}s: {type(e).__name__}: {e}\n")
            continue

        dt = time.time() - t0
        if description is None:
            non_informative += 1
            print(f"    -> NO_CONTENT (non-informative / decorative) [{dt:.1f}s]\n")
        else:
            described += 1
            print(f"    -> description [{dt:.1f}s]:")
            for line in description.splitlines():
                print(f"       {line}")
            print()

    print("-" * 70)
    print(
        f"Summary: {described} described, {non_informative} non-informative, "
        f"{failures} failed (of {min(args.max, len(images))} attempted)."
    )
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
