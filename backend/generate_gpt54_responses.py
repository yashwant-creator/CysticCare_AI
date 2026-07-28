#!/usr/bin/env python3
"""
Generate GPT-5.4 (web-search-enabled) baseline answers for the CysticCare benchmark.

For every *Standard factual* question in the benchmark JSON this script calls the
OpenAI Responses API with the built-in `web_search` tool on model `gpt-5.4`, then
records two fields back onto the same record:

  - "gpt5.4_response": the model's answer text
  - "gpt5.4_sources": list of {"url", "title"} the model cited (deduplicated)

Only Standard factual rows are touched; all other rows are left untouched.

Resumable: rows that already have a non-empty, non-error gpt5.4_response are skipped,
so you can re-run after an interruption. Writes checkpoints to the JSON as it goes.

Run (uses a venv that has the `openai` package installed):
    /path/to/venv/bin/python backend/generate_gpt54_responses.py \
        --json "benchmark result - benchmark result.json" \
        --workers 8
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from openai import OpenAI

MODEL = "gpt-5.4"
SYSTEM = (
    "You are a medical information assistant answering a patient's question about "
    "polycystic kidney disease (PKD/ADPKD). Search the web for authoritative, up-to-date "
    "sources and give an accurate, self-contained answer with inline citations. "
    "Do NOT narrate your search process; output only the final answer to the patient."
)
MAX_OUTPUT_TOKENS = 4000
MAX_RETRIES = 5

RESP_KEY = "gpt5.4_response"
SRC_KEY = "gpt5.4_sources"

_write_lock = threading.Lock()
_done = 0


def load_env(path):
    d = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                d[k.strip()] = v.strip().strip('"').strip("'")
    return d


def clean_url(url):
    """Drop OpenAI's utm_source=openai tracking param."""
    try:
        parts = urlsplit(url)
        q = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if k.lower() != "utm_source"]
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(q), parts.fragment))
    except Exception:
        return url


def extract_sources(resp):
    seen, out = set(), []
    for item in resp.output:
        if getattr(item, "type", None) != "message":
            continue
        for c in getattr(item, "content", []) or []:
            for ann in getattr(c, "annotations", []) or []:
                if getattr(ann, "type", None) == "url_citation":
                    url = clean_url(getattr(ann, "url", "") or "")
                    if url and url not in seen:
                        seen.add(url)
                        out.append({"url": url, "title": getattr(ann, "title", "") or ""})
    return out


def answer_one(client, question):
    """Call gpt-5.4 with web search; return (text, sources). Retries on transient errors."""
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.responses.create(
                model=MODEL,
                instructions=SYSTEM,
                input=question,
                tools=[{"type": "web_search"}],
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
            text = (resp.output_text or "").strip()
            sources = extract_sources(resp)
            if not text:
                raise RuntimeError("empty output_text")
            return text, sources
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = min(60, 2 ** attempt)
            time.sleep(wait)
    raise last_err


def checkpoint(path, records):
    with _write_lock:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)


def needs_work(rec):
    if rec.get("type") != "Standard factual":
        return False
    val = (rec.get(RESP_KEY) or "").strip()
    return (not val) or val.startswith("ERROR:")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--env", default=os.path.join(os.path.dirname(__file__), ".env"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="process at most N (0 = all)")
    ap.add_argument("--checkpoint-every", type=int, default=10)
    args = ap.parse_args()

    env = load_env(args.env)
    key = env.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("No OPENAI_API_KEY found")
    client = OpenAI(api_key=key)

    with open(args.json) as f:
        records = json.load(f)

    todo = [i for i, r in enumerate(records) if needs_work(r)]
    if args.limit:
        todo = todo[: args.limit]
    total = len(todo)
    already = sum(1 for r in records if r.get("type") == "Standard factual") - total
    print(f"START model={MODEL} workers={args.workers} todo={total} "
          f"(already_filled={already}) total_std_factual="
          f"{sum(1 for r in records if r.get('type')=='Standard factual')}", flush=True)
    if total == 0:
        print("ALL DONE nothing to do", flush=True)
        return

    global _done
    errors = 0

    def worker(idx):
        rec = records[idx]
        try:
            text, sources = answer_one(client, rec["question"])
            rec[RESP_KEY] = text
            rec[SRC_KEY] = sources
            return idx, len(sources), None
        except Exception as e:  # noqa: BLE001
            rec[RESP_KEY] = f"ERROR: {type(e).__name__}: {e}"
            rec[SRC_KEY] = []
            return idx, 0, str(e)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(worker, i): i for i in todo}
        for fut in as_completed(futs):
            idx, nsrc, err = fut.result()
            _done += 1
            if err:
                errors += 1
                print(f"[err {_done}/{total}] idx={idx} {err[:120]}", flush=True)
            else:
                q = records[idx]["question"][:60].replace("\n", " ")
                print(f"[ok {_done}/{total}] srcs={nsrc} | {q}", flush=True)
            if _done % args.checkpoint_every == 0:
                checkpoint(args.json, records)

    checkpoint(args.json, records)
    filled = sum(1 for r in records if r.get("type") == "Standard factual"
                 and (r.get(RESP_KEY) or "").strip() and not (r.get(RESP_KEY) or "").startswith("ERROR:"))
    print(f"ALL DONE processed={total} errors={errors} "
          f"std_factual_filled={filled}/162", flush=True)


if __name__ == "__main__":
    main()
