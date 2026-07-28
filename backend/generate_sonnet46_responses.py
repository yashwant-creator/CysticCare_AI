#!/usr/bin/env python3
"""
Generate Claude Sonnet 4.6 (web-search-enabled) baseline answers for the CysticCare benchmark.

For every *Standard factual* question in the benchmark JSON this script calls the
Anthropic Messages API with the built-in `web_search_20250305` server tool
(max_uses=2) on model `claude-sonnet-4-6`, then records two fields onto the record:

  - "sonnet4.6_response": the model's answer text
  - "sonnet4.6_sources": list of {"url", "title"} the model cited (deduplicated)

Only Standard factual rows are touched; all other rows are left untouched.

Resumable: rows that already have a non-empty, non-error sonnet4.6_response are skipped.
Checkpoints to the JSON as it goes.

Run (venv that has the `anthropic` package installed):
    /path/to/venv/bin/python backend/generate_sonnet46_responses.py \
        --json "benchmark result - benchmark result.json" --workers 5
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import anthropic

MODEL = "claude-sonnet-4-6"
SYSTEM = (
    "You are a medical information assistant answering a patient's question about "
    "polycystic kidney disease (PKD/ADPKD). Use web search to find authoritative, up-to-date "
    "sources, then give an accurate, self-contained answer with inline citations. "
    "Do NOT narrate your search process; output only the final answer to the patient."
)
WEB_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 2}
MAX_TOKENS = 4000
MAX_RETRIES = 5
MAX_PAUSE_RESUMES = 4

RESP_KEY = "sonnet4.6_response"
SRC_KEY = "sonnet4.6_sources"

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
    try:
        p = urlsplit(url)
        q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
             if k.lower() != "utm_source"]
        return urlunsplit((p.scheme, p.netloc, p.path, urlencode(q), p.fragment))
    except Exception:
        return url


def parse_content(blocks, text_parts, cited, found):
    """Accumulate answer text, in-text citations, and fallback search-result URLs."""
    for b in blocks:
        bt = getattr(b, "type", None)
        if bt == "text":
            text_parts.append(b.text)
            for cit in (getattr(b, "citations", None) or []):
                u = getattr(cit, "url", None)
                if u:
                    cited.append((clean_url(u), getattr(cit, "title", "") or ""))
        elif bt == "web_search_tool_result":
            content = getattr(b, "content", None)
            if isinstance(content, list):
                for res in content:
                    u = getattr(res, "url", None)
                    if u:
                        found.append((clean_url(u), getattr(res, "title", "") or ""))


def dedup(pairs):
    seen, out = set(), []
    for u, t in pairs:
        if u and u not in seen:
            seen.add(u)
            out.append({"url": u, "title": t})
    return out


def answer_one(client, question):
    """Call Sonnet 4.6 w/ web search; handle pause_turn; return (text, sources)."""
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            messages = [{"role": "user", "content": question}]
            text_parts, cited, found = [], [], []
            resumes = 0
            while True:
                msg = client.messages.create(
                    model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM,
                    tools=[WEB_TOOL], messages=messages,
                )
                parse_content(msg.content, text_parts, cited, found)
                if msg.stop_reason == "pause_turn" and resumes < MAX_PAUSE_RESUMES:
                    resumes += 1
                    messages.append({"role": "assistant", "content": msg.content})
                    continue
                break
            text = "".join(text_parts).strip()
            sources = dedup(cited) or dedup(found)
            if not text:
                raise RuntimeError("empty answer text")
            return text, sources
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(min(60, 2 ** attempt))
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
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    args = ap.parse_args()

    env = load_env(args.env)
    key = env.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        sys.exit("No ANTHROPIC_API_KEY found")
    client = anthropic.Anthropic(api_key=key)

    with open(args.json) as f:
        records = json.load(f)

    todo = [i for i, r in enumerate(records) if needs_work(r)]
    if args.limit:
        todo = todo[: args.limit]
    total = len(todo)
    sf = sum(1 for r in records if r.get("type") == "Standard factual")
    print(f"START model={MODEL} workers={args.workers} todo={total} std_factual={sf}", flush=True)
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
    print(f"ALL DONE processed={total} errors={errors} std_factual_filled={filled}/162", flush=True)


if __name__ == "__main__":
    main()
