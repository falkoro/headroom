"""``RequestOutcome``: the canonical value type for "what happened during
one completed proxy request."

Per the P0 audit (``docs/superpowers/specs/P0-proxy-pipeline-audit.md``),
18 ``metrics.record_request`` call sites across four handler files
disagreed on argument shape — 9 of 18 omitted ``cached=``, 7 of 18
omitted ``attempted_input_tokens=``, only 4 sites emitted a structured
PERF log at all. This module is the structural fix: every handler
converges on building a :class:`RequestOutcome` at end-of-request and
hands it to :func:`emit_request_outcome` (also exposed as
:meth:`HeadroomProxy._record_request_outcome`), which owns the four
downstream effects (Prometheus, cost tracker, request logger, PERF
log).

Note: this is **output unification, not input unification**. Provider
APIs (Anthropic ``/v1/messages``, OpenAI Responses WS, Gemini
``generateContent``, Bedrock, Vertex) stay wildly different — the proxy
talks each upstream in its native dialect. This dataclass standardises
only the *observation* about a completed request. Provider-specific
concepts (Anthropic's 5m/1h cache TTL splits, OpenAI's
inferred-write flag, Gemini's read-only cache count) live as optional
fields with neutral defaults; handlers populate what their provider
actually reports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger("headroom.proxy")


@dataclass(frozen=True)
class RequestOutcome:
    """Immutable, value-equal snapshot of a completed request.

    Construction policy: every field that downstream consumers read MUST
    be either required (no default) or have a neutral default that makes
    the consumer's behaviour identical to "field not present". This keeps
    the contract honest — a handler that forgets a field doesn't silently
    produce wrong metrics; it produces zeros, which the dashboard can
    surface as a missing-data condition (P3 follow-up).
    """

    # ── Identity ──────────────────────────────────────────────────────
    request_id: str
    provider: str
    model: str

    # ── Tokens (required — every site has these) ──────────────────────
    # original_tokens: pre-compression request size, for `tok_before`
    # optimized_tokens: post-compression bytes actually forwarded, for
    #     ``input_tokens`` and ``tok_after``
    # output_tokens: response tokens from upstream
    # tokens_saved: original - optimized (or 0 if compression bypassed)
    # attempted_input_tokens: denominator for active-savings-percent.
    #     The compressible portion only — excludes user messages, system
    #     prompts, prior assistant turns, frozen prefix bytes. This is the
    #     field 7 of 18 audit sites forgot to pass, collapsing
    #     ``active_savings_percent`` to 0 (#454 / #455).
    original_tokens: int
    optimized_tokens: int
    output_tokens: int
    tokens_saved: int
    attempted_input_tokens: int

    # ── Cache (provider-agnostic; unused fields stay 0) ───────────────
    # Anthropic populates all five (read + write + 5m + 1h + uncached).
    # OpenAI populates read + inferred-write + uncached, and sets
    # ``cache_inferred=True`` so the dashboard can warn that the write
    # column is an estimate rather than an upstream-reported counter.
    # Gemini populates read only.
    # Bedrock mirrors Anthropic (it forwards Anthropic-shape usage).
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cache_write_5m_tokens: int = 0
    cache_write_1h_tokens: int = 0
    uncached_input_tokens: int = 0
    cache_inferred: bool = False
    # Response-cache hit (Headroom's own semantic cache served the
    # response from a prior call — completely distinct from
    # upstream-prompt-cache `cache_read_tokens`). True means the proxy
    # never reached the provider at all. Used to drive the
    # Prometheus ``cached`` counter and dashboard "response cache" row.
    from_response_cache: bool = False

    # ── Timing ────────────────────────────────────────────────────────
    # total_latency_ms: wall-clock end-to-end for this request
    # overhead_ms: time spent in compression dispatch only (subset of total)
    # ttfb_ms: time to first upstream byte for streaming paths; 0 for
    #     non-streaming or when unmeasured (no None — convention is 0)
    # pipeline_timing: optional per-stage breakdown surfaced on dashboards
    total_latency_ms: float = 0.0
    overhead_ms: float = 0.0
    ttfb_ms: float = 0.0
    pipeline_timing: dict[str, float] | None = None

    # ── Transforms + diagnostics ──────────────────────────────────────
    # transforms_applied: tuple (immutable) of every transform that ran.
    #     RequestLog still wants list[str]; the funnel converts at the
    #     boundary.
    # waste_signals: per-router signals captured during routing (counts
    #     of skipped vs applied units etc.); dashboards summarise.
    # num_messages: messages in the original request (for ``msgs=N`` in
    #     PERF), counted from body.input/body.messages.
    # turn_id: stable hash of the conversation prefix; used by
    #     dashboards to group multi-turn sessions.
    # request_messages: only populated when ``config.log_full_messages``
    #     is enabled (off by default — message bodies are sensitive).
    # tags: client-provided routing/identification tags.
    transforms_applied: tuple[str, ...] = ()
    waste_signals: dict[str, int] | None = None
    num_messages: int = 0
    turn_id: str | None = None
    request_messages: list[dict[str, Any]] | None = None
    tags: dict[str, str] = field(default_factory=dict)

    # ── Derived (computed once, no caching needed — properties are cheap) ─

    @property
    def cache_hit(self) -> bool:
        """True iff EITHER upstream reported a cache read OR the response
        was served from Headroom's own response cache.

        Two distinct concepts collapsed into one observable boolean for
        downstream consumers (Prometheus ``cached`` counter, RequestLog
        ``cache_hit`` flag). The dataclass tracks them separately so
        dashboards can split them; the derived property unifies them.

        Pre-refactor 9 of 18 sites hardcoded this to False — this property
        makes "I forgot to compute it" structurally impossible.
        """
        return self.cache_read_tokens > 0 or self.from_response_cache

    @property
    def cache_hit_pct(self) -> int:
        """Cache read share of (read + write), rounded to int percent.

        Returns 0 when neither read nor write fired (a request that did no
        cache work; distinguishing this from "0% hit rate on real cache
        work" requires looking at the absolute values, not the ratio).
        """
        denom = self.cache_read_tokens + self.cache_write_tokens
        if denom <= 0:
            return 0
        return round(self.cache_read_tokens / denom * 100)

    @property
    def savings_pct(self) -> float:
        """Compression savings as a fraction of the original request size.

        This is the proxy-side ratio: ``tokens_saved / original_tokens``.
        The dashboard headline "active savings percent" uses a different
        ratio (``tokens_saved / attempted_input_tokens``) — see the
        Prometheus metric for the active calculation.
        """
        if self.original_tokens <= 0:
            return 0.0
        return self.tokens_saved / self.original_tokens * 100.0


# ── The funnel ───────────────────────────────────────────────────────


async def emit_request_outcome(handler: Any, outcome: RequestOutcome) -> None:
    """Single funnel for per-request bookkeeping. The contract.

    Owns the four downstream effects in canonical order:

      1. ``handler.metrics.record_request(...)`` — Prometheus / SavingsTracker
      2. ``handler.cost_tracker.record_tokens(...)`` — cost dashboard
         (skipped when cost_tracker is None, i.e. ``--no-cost``)
      3. ``handler.logger.log(RequestLog(...))`` — per-request log feed
         (skipped when logger is None, i.e. ``--no-request-logging``)
      4. structured PERF log line — consumed by ``headroom perf``

    Takes the handler as a free argument rather than ``self`` so this
    function is callable from:
    * ``HeadroomProxy._record_request_outcome`` (production)
    * any test dummy that has the three required attributes
      (``metrics``, ``cost_tracker``, optionally ``logger``)
    * any provider handler mixin

    The handler argument is structurally typed (duck-typed); no formal
    Protocol — the requirement is simply that ``handler.metrics`` exists
    and is awaitable-compatible. We could lift this to a typing.Protocol
    if/when another contract surface emerges, but YAGNI.
    """
    from headroom.proxy.cost import _summarize_transforms
    from headroom.proxy.models import RequestLog

    # 1. Prometheus / SavingsTracker.
    await handler.metrics.record_request(
        provider=outcome.provider,
        model=outcome.model,
        input_tokens=outcome.optimized_tokens,
        output_tokens=outcome.output_tokens,
        tokens_saved=outcome.tokens_saved,
        latency_ms=outcome.total_latency_ms,
        cached=outcome.cache_hit,
        overhead_ms=outcome.overhead_ms,
        ttfb_ms=outcome.ttfb_ms,
        pipeline_timing=outcome.pipeline_timing,
        waste_signals=outcome.waste_signals,
        cache_read_tokens=outcome.cache_read_tokens,
        cache_write_tokens=outcome.cache_write_tokens,
        cache_write_5m_tokens=outcome.cache_write_5m_tokens,
        cache_write_1h_tokens=outcome.cache_write_1h_tokens,
        uncached_input_tokens=outcome.uncached_input_tokens,
        attempted_input_tokens=outcome.attempted_input_tokens,
    )

    # 2. Cost tracker (optional).
    cost_tracker = getattr(handler, "cost_tracker", None)
    if cost_tracker is not None:
        cost_tracker.record_tokens(
            outcome.model,
            outcome.tokens_saved,
            outcome.optimized_tokens,
            cache_read_tokens=outcome.cache_read_tokens,
            cache_write_tokens=outcome.cache_write_tokens,
            cache_write_5m_tokens=outcome.cache_write_5m_tokens,
            cache_write_1h_tokens=outcome.cache_write_1h_tokens,
            uncached_tokens=outcome.uncached_input_tokens,
        )

    # 3. Per-request log (optional).
    request_logger = getattr(handler, "logger", None)
    if request_logger is not None:
        request_logger.log(
            RequestLog(
                request_id=outcome.request_id,
                timestamp=datetime.now().isoformat(),
                provider=outcome.provider,
                model=outcome.model,
                input_tokens_original=outcome.original_tokens,
                input_tokens_optimized=outcome.optimized_tokens,
                output_tokens=outcome.output_tokens,
                tokens_saved=outcome.tokens_saved,
                savings_percent=outcome.savings_pct,
                optimization_latency_ms=outcome.overhead_ms,
                total_latency_ms=outcome.total_latency_ms,
                tags=outcome.tags,
                cache_hit=outcome.cache_hit,
                transforms_applied=list(outcome.transforms_applied),
                waste_signals=outcome.waste_signals,
                request_messages=outcome.request_messages,
                turn_id=outcome.turn_id,
            )
        )

    # 4. Structured PERF log line.
    logger.info(
        f"[{outcome.request_id}] PERF "
        f"model={outcome.model} msgs={outcome.num_messages} "
        f"tok_before={outcome.original_tokens} tok_after={outcome.optimized_tokens} "
        f"tok_saved={outcome.tokens_saved} "
        f"cache_read={outcome.cache_read_tokens} cache_write={outcome.cache_write_tokens} "
        f"cache_hit_pct={outcome.cache_hit_pct} "
        f"opt_ms={outcome.overhead_ms:.0f} "
        f"transforms={_summarize_transforms(list(outcome.transforms_applied))}"
    )
