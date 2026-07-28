#!/usr/bin/env python3
"""
Augment pipeline_results.json with the live CysticCare RAG bot's answer + metadata.

For every question in the input JSON this script calls the configured CysticCare
``/chat`` endpoint with one standard-RAG retrieval path (query suffixed with
" for ADPKD", adaptive routing/CoT/Stepback disabled) and appends two new fields
to each record:

  - "cysticcare_response": the bot's answer text
  - "cysticcare_metadata": the sources/citations + validation + follow-ups the bot used

The original file is left untouched; results are written to a copy.

Usage:
    python3 generate_cysticcare_responses.py \
        [--input  backend/app/pipeline_results.json] \
        [--output backend/app/pipeline_results_with_cysticcare.json] \
        [--workers 5] [--limit N] [--force]

Re-running resumes by default: questions already answered in the output file
are skipped. Pass ``--force`` after rebuilding or retuning retrieval so every
row is generated against the current index.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

BACKEND_URL = os.environ.get(
    "CYSTICCARE_BACKEND_URL",
    "https://cysticcare-backend-798621865464.us-east1.run.app",
).rstrip("/")
CHAT_ENDPOINT = BACKEND_URL + "/chat"

# How the Flutter app phrases queries (lib/services/backend_api.dart)
QUERY_SUFFIX = " for ADPKD"

REQUEST_TIMEOUT = 300  # seconds per request; matches Cloud Run's own 300s timeout
                       # so we never abandon a request the backend would complete
MAX_RETRIES = 3
RETRY_BACKOFF = 5  # seconds, multiplied by attempt number

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(HERE, "app", "pipeline_results.json")
DEFAULT_OUTPUT = os.path.join(HERE, "app", "pipeline_results_with_cysticcare.json")


_HTTP_MARKER = "\n__HTTP_CODE__"


def call_chat(question: str, session_id: str) -> dict:
    """POST one question to /chat and return the parsed response, with retries.

    Uses curl with a hard --max-time ceiling (plus a subprocess timeout as a
    second safety net) so a hung connection can never block a worker forever.
    """
    payload = json.dumps(
        {
            "query": question + QUERY_SUFFIX,
            "session_id": session_id,
            # Retrieval metrics require one ranked list for the original
            # query. CoT and Stepback combine separate query rankings, which
            # are not a meaningful single rank order for MRR/nDCG.
            "use_adaptive_agent": False,
            "use_stepback": False,
            "use_cot": False,
            "pre_check_topic": True,
            # Preserve stable paper/chunk IDs for retrieval evaluation. This
            # is an offline benchmark flag; the Flutter client leaves it off.
            "include_retrieval_debug": True,
        }
    )

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            proc = subprocess.run(
                [
                    "curl", "-s", "-S",
                    "--max-time", str(REQUEST_TIMEOUT),
                    "--connect-timeout", "20",
                    "-X", "POST", CHAT_ENDPOINT,
                    "-H", "Content-Type: application/json",
                    "-d", payload,
                    "-w", _HTTP_MARKER + "%{http_code}",
                ],
                capture_output=True,
                text=True,
                timeout=REQUEST_TIMEOUT + 30,  # hard kill if curl itself wedges
            )
            out = proc.stdout or ""
            marker = out.rfind(_HTTP_MARKER)
            if marker == -1:
                raise RuntimeError(
                    f"curl rc={proc.returncode}: {(proc.stderr or '')[:300]}"
                )
            body = out[:marker]
            code = out[marker + len(_HTTP_MARKER):].strip()
            if code != "200":
                raise RuntimeError(f"HTTP {code}: {body[:300]}")
            return json.loads(body)
        except subprocess.TimeoutExpired:
            last_err = f"curl timed out after {REQUEST_TIMEOUT}s"
        except Exception as e:  # noqa: BLE001 - report any transport/parse error
            last_err = f"{type(e).__name__}: {e}"

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF * attempt)

    raise RuntimeError(last_err or "unknown error")


def build_metadata(data: dict) -> dict:
    """Extract the metadata the bot used to answer (sources/citations + extras)."""
    return {
        "sources": data.get("sources", []),
        "validation": data.get("validation"),
        "followup_questions": data.get("followup_questions", []),
        "stepback_query": data.get("stepback_query", ""),
        "cot_enabled": data.get("cot_enabled", False),
        "status": data.get("status"),
        "retrieval_metadata": data.get("retrieval_metadata", {}),
        "retrieved_chunks": data.get("retrieved_chunks", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="process at most N questions (0 = all)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="ignore prior output and regenerate every selected question",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Resume support: reuse already-computed answers from a previous output file.
    if os.path.exists(args.output) and not args.force:
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                prior = json.load(f)
            by_q = {r.get("question"): r for r in prior}
            for rec in records:
                prev = by_q.get(rec.get("question"))
                if prev and prev.get("cysticcare_response") is not None:
                    rec["cysticcare_response"] = prev["cysticcare_response"]
                    rec["cysticcare_metadata"] = prev.get("cysticcare_metadata")
            print(f"Resuming: loaded prior results from {args.output}")
        except Exception as e:  # noqa: BLE001
            print(f"Could not reuse existing output ({e}); starting fresh.")
    elif args.force:
        print("Force mode: ignoring prior output and regenerating selected questions.")

    todo = [
        i
        for i, rec in enumerate(records)
        if rec.get("cysticcare_response") is None
    ]
    if args.limit:
        todo = todo[: args.limit]

    total = len(records)
    print(f"Backend : {CHAT_ENDPOINT}")
    print(f"Records : {total} total, {len(todo)} to process, workers={args.workers}")
    if not todo:
        print("Nothing to do.")
        _write(records, args.output)
        return 0

    lock = threading.Lock()
    done = {"n": 0, "ok": 0, "err": 0}
    start = time.time()

    def worker(idx: int):
        rec = records[idx]
        question = rec["question"]
        session_id = f"pipeline_eval_{idx}"
        try:
            data = call_chat(question, session_id)
            rec["cysticcare_response"] = data.get("response", "")
            rec["cysticcare_metadata"] = build_metadata(data)
            status = "ok"
        except Exception as e:  # noqa: BLE001
            rec["cysticcare_response"] = None
            rec["cysticcare_metadata"] = {"error": str(e)}
            status = "err"
        with lock:
            done["n"] += 1
            done[status] += 1
            n = done["n"]
            elapsed = time.time() - start
            rate = n / elapsed if elapsed else 0
            eta = (len(todo) - n) / rate if rate else 0
            tag = "OK " if status == "ok" else "ERR"
            print(
                f"[{n}/{len(todo)}] {tag} q#{idx} "
                f"({done['ok']} ok, {done['err']} err) "
                f"ETA {eta/60:.1f}m | {question[:60]}...",
                flush=True,
            )
            # Checkpoint after every completion so progress is visible live and
            # survives interruption (the file write is cheap and atomic).
            _write(records, args.output)
        return status

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, idx) for idx in todo]
        for _ in as_completed(futures):
            pass

    _write(records, args.output)
    elapsed = time.time() - start
    print(
        f"\nDone in {elapsed/60:.1f}m. "
        f"{done['ok']} succeeded, {done['err']} failed. "
        f"Output: {args.output}"
    )
    return 0 if done["err"] == 0 else 1


def _write(records, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


if __name__ == "__main__":
    sys.exit(main())
