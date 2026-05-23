import json
import logging
import os
from contextlib import ExitStack, contextmanager
from functools import wraps
from typing import Any, Dict, Iterator, Optional


logger = logging.getLogger(__name__)

_TRACE_STATE: Dict[str, Any] = {
    "initialized": False,
    "enabled": False,
    "tracer": None,
    "using_metadata": None,
    "using_session": None,
    "using_user": None,
}

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _parse_bool_env(var_name: str) -> Optional[bool]:
    value = os.getenv(var_name)
    if value is None:
        return None
    return value.strip().lower() in _TRUE_VALUES


def _should_enable_traceai() -> bool:
    explicit_flag = _parse_bool_env("TRACEAI_ENABLED")
    if explicit_flag is not None:
        return explicit_flag
    return bool(os.getenv("FI_API_KEY") and os.getenv("FI_SECRET_KEY"))


def _normalize_attribute_value(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, (list, tuple)):
        if all(isinstance(item, (bool, int, float, str)) for item in value):
            return list(value)
        return json.dumps(value, default=str)

    return json.dumps(value, default=str)


def initialize_traceai() -> bool:
    if _TRACE_STATE["initialized"]:
        return _TRACE_STATE["enabled"]

    _TRACE_STATE["initialized"] = True

    if not _should_enable_traceai():
        logger.info("traceAI is disabled; install optional traceAI deps and set Future AGI credentials to enable it")
        return False

    if not os.getenv("FI_API_KEY") or not os.getenv("FI_SECRET_KEY"):
        logger.warning("traceAI requested but FI_API_KEY/FI_SECRET_KEY are not configured")
        return False

    try:
        from fi_instrumentation import FITracer, register, using_metadata, using_session, using_user
        from fi_instrumentation.fi_types import ProjectType
        from traceai_openai import OpenAIInstrumentor
    except ImportError as exc:
        logger.warning("traceAI requested but optional Python packages are unavailable: %s", exc)
        return False

    project_name = os.getenv("TRACEAI_PROJECT_NAME", "CysticCare AI")
    project_version = os.getenv("TRACEAI_PROJECT_VERSION")
    project_type_name = os.getenv("TRACEAI_PROJECT_TYPE", "OBSERVE").upper()
    project_type = getattr(ProjectType, project_type_name, ProjectType.OBSERVE)

    if project_type_name == "OBSERVE" and project_version:
        logger.info("Ignoring TRACEAI_PROJECT_VERSION for OBSERVE projects")
        project_version = None

    register_kwargs = {
        "project_type": project_type,
        "project_name": project_name,
        "set_global_tracer_provider": True,
    }
    if project_version:
        register_kwargs["project_version_name"] = project_version

    try:
        trace_provider = register(**register_kwargs)
        OpenAIInstrumentor().instrument(tracer_provider=trace_provider)
    except Exception as exc:
        logger.warning("traceAI initialization failed: %s", exc)
        return False

    _TRACE_STATE.update(
        enabled=True,
        tracer=FITracer(trace_provider.get_tracer(__name__)),
        using_metadata=using_metadata,
        using_session=using_session,
        using_user=using_user,
    )

    logger.info(
        "traceAI initialized for project=%s%s",
        project_name,
        f", version={project_version}" if project_version else "",
    )
    return True


def set_span_attributes(span: Any, attributes: Optional[Dict[str, Any]]) -> None:
    if span is None or not attributes:
        return

    for key, value in attributes.items():
        normalized = _normalize_attribute_value(value)
        if normalized is None:
            continue
        try:
            span.set_attribute(key, normalized)
        except Exception:
            logger.debug("Failed to set trace attribute %s", key, exc_info=True)


def set_span_output(span: Any, output_value: Any) -> None:
    if span is None or output_value is None or not hasattr(span, "set_output"):
        return

    try:
        span.set_output(output_value)
    except Exception:
        logger.debug("Failed to set trace span output", exc_info=True)


@contextmanager
def trace_context(
    *,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Iterator[None]:
    if not initialize_traceai() or _TRACE_STATE["tracer"] is None:
        yield
        return

    with ExitStack() as stack:
        if session_id:
            stack.enter_context(_TRACE_STATE["using_session"](str(session_id)))
        if user_id:
            stack.enter_context(_TRACE_STATE["using_user"](str(user_id)))
        if metadata:
            stack.enter_context(_TRACE_STATE["using_metadata"](metadata))
        yield


@contextmanager
def trace_span(
    name: str,
    *,
    kind: str = "chain",
    attributes: Optional[Dict[str, Any]] = None,
    input_value: Any = None,
) -> Iterator[Any]:
    if not initialize_traceai() or _TRACE_STATE["tracer"] is None:
        yield None
        return

    with _TRACE_STATE["tracer"].start_as_current_span(name, fi_span_kind=kind) as span:
        if input_value is not None and hasattr(span, "set_input"):
            try:
                span.set_input(input_value)
            except Exception:
                logger.debug("Failed to set trace span input", exc_info=True)

        set_span_attributes(span, attributes)

        try:
            yield span
        except Exception as exc:
            if hasattr(span, "record_exception"):
                span.record_exception(exc)
            set_span_attributes(
                span,
                {
                    "error.type": exc.__class__.__name__,
                    "error.message": str(exc),
                },
            )
            raise


def _is_wrapped(callable_obj: Any) -> bool:
    return bool(getattr(callable_obj, "__traceai_wrapped__", False))


def _mark_wrapped(callable_obj: Any) -> Any:
    callable_obj.__traceai_wrapped__ = True
    return callable_obj


def _infer_agent_type(request: Any) -> str:
    if getattr(request, "use_stepback", False):
        return "stepback"
    if getattr(request, "use_cot", False):
        return "cot"
    return "standard_rag"


def _replace_fastapi_endpoint(app: Any, original: Any, wrapped: Any) -> None:
    for route in getattr(app, "routes", []):
        if getattr(route, "endpoint", None) is not original:
            continue

        route.endpoint = wrapped
        if hasattr(route, "dependant"):
            route.dependant.call = wrapped


def _patch_chat_endpoint(main_module: Any) -> None:
    original = getattr(main_module, "chat_endpoint", None)
    if original is None or _is_wrapped(original):
        return

    @wraps(original)
    async def wrapped(request: Any) -> Any:
        with trace_context(
            session_id=getattr(request, "session_id", None),
            metadata={
                "route": "/chat",
                "top_k": getattr(request, "top_k", None),
                "use_adaptive_agent": getattr(request, "use_adaptive_agent", None),
                "use_cot": getattr(request, "use_cot", None),
                "use_stepback": getattr(request, "use_stepback", None),
                "use_validation": getattr(request, "use_validation", None),
                "use_query_rewriting": getattr(request, "use_query_rewriting", None),
            },
        ):
            with trace_span(
                "rag.chat_request",
                kind="agent",
                input_value=getattr(request, "query", None),
                attributes={
                    "rag.session_id": getattr(request, "session_id", None),
                    "rag.top_k": getattr(request, "top_k", None),
                    "rag.temperature": getattr(request, "temperature", None),
                    "rag.max_tokens": getattr(request, "max_tokens", None),
                },
            ) as request_span:
                try:
                    response = await original(request)
                except Exception as exc:
                    status_code = getattr(exc, "status_code", None)
                    set_span_attributes(
                        request_span,
                        {
                            "rag.status": "error",
                            "http.status_code": status_code,
                        },
                    )
                    raise

                response_text = getattr(response, "response", "")
                sources = getattr(response, "sources", []) or []
                followups = getattr(response, "followup_questions", []) or []
                validation = getattr(response, "validation", None)
                refused = bool(main_module.is_refusal_response(response_text))

                attributes = {
                    "rag.status": getattr(response, "status", "success"),
                    "rag.agent_type": _infer_agent_type(request),
                    "rag.refused": refused,
                    "rag.source_count": len(sources),
                    "rag.followup_count": len(followups),
                }
                if validation is not None:
                    attributes["validation.passed"] = getattr(validation, "passed", None)
                    attributes["validation.overall_score"] = getattr(validation, "overall_score", None)
                    attributes["validation.was_regenerated"] = getattr(validation, "was_regenerated", None)

                set_span_attributes(request_span, attributes)
                set_span_output(
                    request_span,
                    {
                        "status": getattr(response, "status", "success"),
                        "refused": refused,
                        "source_count": len(sources),
                        "followup_count": len(followups),
                    },
                )
                return response

    wrapped = _mark_wrapped(wrapped)
    main_module.chat_endpoint = wrapped
    if hasattr(main_module, "app"):
        _replace_fastapi_endpoint(main_module.app, original, wrapped)


def _patch_followup_agent(followup_module: Any) -> None:
    cls = getattr(followup_module, "FollowUpAgent", None)
    original = getattr(cls, "generate_followup_questions", None)
    if cls is None or original is None or _is_wrapped(original):
        return

    @wraps(original)
    async def wrapped(self: Any, query: str, response: str, *args: Any, **kwargs: Any) -> Any:
        with trace_span(
            "rag.followup_questions",
            kind="agent",
            input_value=query,
        ) as followup_span:
            questions = await original(self, query, response, *args, **kwargs)
            set_span_attributes(
                followup_span,
                {
                    "rag.status": "success",
                    "rag.followup_count": len(questions or []),
                },
            )
            return questions

    setattr(cls, "generate_followup_questions", _mark_wrapped(wrapped))


def _patch_validation_agent(validation_module: Any) -> None:
    cls = getattr(validation_module, "ValidationAgent", None)
    original = getattr(cls, "validate_and_retry", None)
    if cls is None or original is None or _is_wrapped(original):
        return

    @wraps(original)
    async def wrapped(
        self: Any,
        query: str,
        answer: str,
        retrieved_chunks: Any,
        agent_type: str = "standard_rag",
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        with trace_span(
            "rag.validation",
            kind="agent",
            input_value=query,
            attributes={
                "rag.agent_type": agent_type,
                "rag.retrieved_chunk_count": len(retrieved_chunks or []),
            },
        ) as validation_span:
            result = await original(
                self,
                query,
                answer,
                retrieved_chunks,
                agent_type=agent_type,
                *args,
                **kwargs,
            )

            validation = result.get("validation")
            validation_data = validation.to_dict() if hasattr(validation, "to_dict") else {}
            set_span_attributes(
                validation_span,
                {
                    "rag.status": "success",
                    "validation.passed": validation_data.get("passed"),
                    "validation.overall_score": validation_data.get("overall_score"),
                    "validation.was_regenerated": result.get("was_regenerated"),
                },
            )
            return result

    setattr(cls, "validate_and_retry", _mark_wrapped(wrapped))


def _patch_openai_rag_init(openai_rag_init: Any) -> None:
    original_initialize = getattr(openai_rag_init, "initialize_openai_rag_system", None)
    if original_initialize is not None and not _is_wrapped(original_initialize):
        @wraps(original_initialize)
        async def initialize_wrapper(*args: Any, **kwargs: Any) -> Any:
            pdf_directory = kwargs.get("pdf_directory", args[0] if args else "papers")
            collection_name = kwargs.get("collection_name", "pkd_knowledge_base_openai")
            with trace_span(
                "rag.initialize",
                kind="chain",
                attributes={
                    "rag.pdf_directory": pdf_directory,
                    "rag.collection_name": collection_name,
                },
            ) as init_span:
                result = await original_initialize(*args, **kwargs)
                set_span_attributes(
                    init_span,
                    {
                        "rag.status": result.get("status"),
                        "rag.documents_processed": result.get("documents_processed"),
                        "rag.chunks_created": result.get("chunks_created"),
                        "rag.total_vectors": result.get("total_vectors"),
                    },
                )
                return result

        openai_rag_init.initialize_openai_rag_system = _mark_wrapped(initialize_wrapper)

    original_search = getattr(openai_rag_init, "search_knowledge_base", None)
    if original_search is not None and not _is_wrapped(original_search):
        @wraps(original_search)
        async def search_wrapper(query: str, top_k: int = 5, *args: Any, **kwargs: Any) -> Any:
            with trace_span(
                "rag.search_knowledge_base",
                kind="retriever",
                input_value=query,
                attributes={"rag.top_k": top_k},
            ) as search_span:
                result = await original_search(query, top_k, *args, **kwargs)
                set_span_attributes(
                    search_span,
                    {
                        "rag.status": result.get("status"),
                        "rag.result_count": len(result.get("results", [])),
                        "rag.top_relevance_score": (
                            result.get("results", [{}])[0].get("relevance_score")
                            if result.get("results")
                            else None
                        ),
                    },
                )
                return result

        openai_rag_init.search_knowledge_base = _mark_wrapped(search_wrapper)

    original_rag = getattr(openai_rag_init, "get_rag_response", None)
    if original_rag is not None and not _is_wrapped(original_rag):
        @wraps(original_rag)
        async def rag_wrapper(
            query: str,
            top_k: int = 5,
            temperature: float = 0.7,
            max_tokens: int = 2000,
            use_query_rewriting: bool = True,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            with trace_span(
                "rag.standard_pipeline",
                kind="chain",
                input_value=query,
                attributes={
                    "rag.top_k": top_k,
                    "rag.use_query_rewriting": use_query_rewriting,
                },
            ) as pipeline_span:
                result = await original_rag(
                    query,
                    top_k=top_k,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    use_query_rewriting=use_query_rewriting,
                    *args,
                    **kwargs,
                )
                set_span_attributes(
                    pipeline_span,
                    {
                        "rag.status": result.get("status"),
                        "rag.refused": result.get("refused"),
                        "rag.source_count": len(result.get("sources", [])),
                        "rag.retrieved_chunk_count": len(result.get("retrieved_chunks", [])),
                    },
                )
                return result

        openai_rag_init.get_rag_response = _mark_wrapped(rag_wrapper)


def _patch_cot_service(cot_module: Any) -> None:
    cls = getattr(cot_module, "ChainOfThoughtRAG", None)
    if cls is None:
        return

    original = getattr(cls, "get_cot_rag_response", None)
    if original is not None and not _is_wrapped(original):
        @wraps(original)
        async def pipeline_wrapper(
            self: Any,
            query: str,
            top_k_per_step: int = 5,
            temperature: float = 0.7,
            max_tokens: int = 2000,
            max_steps: int = 5,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            with trace_span(
                "rag.cot_pipeline",
                kind="chain",
                input_value=query,
                attributes={
                    "rag.top_k_per_step": top_k_per_step,
                    "rag.max_steps": max_steps,
                },
            ) as span:
                result = await original(
                    self,
                    query,
                    top_k_per_step=top_k_per_step,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_steps=max_steps,
                    *args,
                    **kwargs,
                )
                set_span_attributes(
                    span,
                    {
                        "rag.status": result.get("status"),
                        "rag.refused": result.get("refused"),
                        "rag.off_topic": result.get("off_topic"),
                        "rag.reasoning_step_count": len(result.get("reasoning_chain", [])),
                        "rag.source_count": len(result.get("sources", [])),
                    },
                )
                return result
        setattr(cls, "get_cot_rag_response", _mark_wrapped(pipeline_wrapper))


def _patch_stepback_service(stepback_module: Any) -> None:
    cls = getattr(stepback_module, "StepbackAgent", None)
    if cls is None:
        return

    original = getattr(cls, "answer_with_stepback", None)
    if original is not None and not _is_wrapped(original):
        @wraps(original)
        async def answer_wrapper(
            self: Any,
            query: str,
            top_k: int = 5,
            temperature: float = 0.7,
            max_tokens: int = 2000,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            with trace_span(
                "rag.stepback_pipeline",
                kind="chain",
                input_value=query,
                attributes={"rag.top_k": top_k},
            ) as span:
                result = await original(
                    self,
                    query,
                    top_k=top_k,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    *args,
                    **kwargs,
                )
                set_span_attributes(
                    span,
                    {
                        "rag.status": result.get("status"),
                        "rag.refused": result.get("refused"),
                        "rag.off_topic": result.get("off_topic"),
                        "rag.source_count": len(result.get("sources", [])),
                        "rag.combined_count": result.get("retrieval_metadata", {}).get("combined_count"),
                    },
                )
                return result
        setattr(cls, "answer_with_stepback", _mark_wrapped(answer_wrapper))


def setup_traceai(main_module: Any = None) -> bool:
    if not initialize_traceai():
        return False

    from ..services import cot_rag_service, followup_agent, openai_rag_init, stepback_agent, validation_agent

    _patch_openai_rag_init(openai_rag_init)
    _patch_cot_service(cot_rag_service)
    _patch_stepback_service(stepback_agent)
    _patch_followup_agent(followup_agent)
    _patch_validation_agent(validation_agent)

    if main_module is not None:
        main_module.initialize_openai_rag_system = openai_rag_init.initialize_openai_rag_system
        main_module.get_rag_response = openai_rag_init.get_rag_response
        _patch_chat_endpoint(main_module)

    return True
