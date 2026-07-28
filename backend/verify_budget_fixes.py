"""
Verify the gpt-5.5 reasoning-model fixes: every agent that previously failed
(either a 400 on custom `temperature` or an empty response from a tiny token
budget) now returns real content from a successful API call.

A check PASSES only if BOTH hold:
  * no WARNING/ERROR was logged during the call (every fallback path logs one), and
  * the returned value passes a content assertion (not a canned fallback).

Makes real (billable) OpenAI calls using OPENAI_API_KEY from backend/.env.
Run from backend/:  .venv/bin/python verify_budget_fixes.py
"""
import asyncio
import logging
import os
import time

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from app.services.openai_service import OpenAIService
from app.utils.openai_utils import load_session_config
from app.utils.prompt_guard import is_question_on_topic
from app.services.query_rewriter import QueryRewriter
from app.services.stepback_agent import StepbackAgent
from app.services.validation_agent import ValidationAgent
from app.services.cot_rag_service import ChainOfThoughtRAG
from app.services.followup_agent import FollowUpAgent


class LogCapture(logging.Handler):
    """Collects WARNING+ records so we can detect silent fallback paths."""
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records = []
    def emit(self, record):
        self.records.append(record)


capture = LogCapture()
logging.getLogger().addHandler(capture)
logging.getLogger().setLevel(logging.INFO)

results = []


async def run_check(name, coro_or_value, validate, render):
    """Run one check; PASS only if no WARN+/ERROR logged AND validate() is True."""
    start = len(capture.records)
    t0 = time.time()
    try:
        value = await coro_or_value if asyncio.iscoroutine(coro_or_value) else coro_or_value
        err = None
    except Exception as e:  # pragma: no cover
        value, err = None, e
    dt = time.time() - t0
    new_logs = capture.records[start:]
    logged_err = [r for r in new_logs if r.levelno >= logging.WARNING]
    ok = err is None and not logged_err and validate(value)
    tag = "PASS ✅" if ok else "FAIL ❌"
    print(f"{tag}  {name}  [{dt:.1f}s]")
    if err:
        print(f"        raised: {type(err).__name__}: {err}")
    for r in logged_err:
        print(f"        LOGGED {r.levelname}: {r.getMessage()[:160]}")
    if value is not None:
        for line in render(value).splitlines():
            print(f"        {line}")
    print()
    results.append((name, ok))


async def main():
    cfg = load_session_config()
    svc = OpenAIService(
        embedding_model=cfg["embedding_model"],
        chat_model=cfg["chat_model"],
        vision_model=cfg["vision_model"],
        max_retries=cfg["max_retries"],
        retry_delay=cfg["retry_delay"],
    )
    print(f"chat model: {cfg['chat_model']}  is_reasoning_model={svc.is_reasoning_model}")
    print("-" * 70)

    # 1) prompt_guard — force the LLM path (no kidney anchor) and require a real
    #    ON/OFF-TOPIC verdict, not the "classifier failure" error fallback.
    async def guard():
        for q in ["Is it safe to take ibuprofen regularly for back pain?",
                  "What dietary changes help slow the progression of my condition?"]:
            res = await is_question_on_topic(q, svc)
            if res.get("source") == "llm":
                res["_query"] = q
                return res
        return {"source": "heuristic"}
    await run_check(
        "prompt_guard.is_question_on_topic (LLM path)", guard(),
        lambda r: r.get("source") == "llm"
        and "failure" not in r.get("classification", "")
        and r.get("classification", "").upper().startswith(("ON-TOPIC", "OFF-TOPIC")),
        lambda r: f'query: "{r.get("_query")}"\n'
                  f'on_topic={r["on_topic"]} classification={r.get("classification")!r}',
    )

    # 2) query_rewriter — real variations parsed from JSON
    await run_check(
        "query_rewriter.rewrite_query_advanced",
        QueryRewriter(svc).rewrite_query_advanced("PKD treatment options"),
        lambda out: len(out.get("variations") or []) > 0,
        lambda out: f'{len(out.get("variations") or [])} variation(s):\n' +
                    "\n".join(f"- {v}" for v in (out.get("variations") or [])[:4]),
    )

    # 3) stepback — not the templated fallback string
    fallback_sb = "What are the general principles and mechanisms related to"
    await run_check(
        "stepback_agent.generate_stepback_query",
        StepbackAgent(svc).generate_stepback_query("Should I avoid caffeine if I have PKD?"),
        lambda sb: bool(sb and sb.strip()) and not sb.startswith(fallback_sb),
        lambda sb: repr(sb),
    )

    # 4) validation._check_relevance — real parsed score (fallback is exactly 0.75)
    await run_check(
        "validation_agent._check_relevance",
        # _check_relevance is sync; wrap result so run_check handles it uniformly
        asyncio.to_thread(
            ValidationAgent(svc)._check_relevance,
            "What is tolvaptan used for in ADPKD?",
            "Tolvaptan is a vasopressin V2-receptor antagonist used to slow kidney "
            "function decline and cyst growth in ADPKD.",
        ),
        lambda chk: chk.score > 0,
        lambda chk: f"score={chk.score} passed={chk.passed}",
    )

    cot = ChainOfThoughtRAG(svc)
    # 5) cot decompose — multiple distinct sub-questions (fallback returns [query])
    await run_check(
        "cot_rag.decompose_query",
        cot.decompose_query("How does PKD progress and how is it treated?"),
        lambda subs: len(subs) >= 2,
        lambda subs: f"{len(subs)} sub-question(s):\n" + "\n".join(f"- {s}" for s in subs[:5]),
    )

    # 6) cot reason step — not the "Unable to reason" error string
    await run_check(
        "cot_rag.reason_through_step",
        cot.reason_through_step(
            "What causes cyst growth in PKD?",
            "In ADPKD, PKD1/PKD2 mutations disrupt polycystin signaling, raising cAMP "
            "and driving fluid secretion and cyst expansion.",
            [],
        ),
        lambda s: bool(s and s.strip()) and "Unable to reason" not in s and "Error code" not in s,
        lambda s: repr(s[:200]),
    )

    # 7) followup — real list of questions
    await run_check(
        "followup_agent.generate_followup_questions",
        FollowUpAgent(svc).generate_followup_questions(
            "What is tolvaptan?", "Tolvaptan is a drug that slows cyst growth in ADPKD."),
        lambda fups: len(fups) > 0,
        lambda fups: f"{len(fups)} question(s):\n" + "\n".join(f"- {f}" for f in fups),
    )

    print("=" * 70)
    passed = sum(1 for _, ok in results if ok)
    print(f"RESULT: {passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
