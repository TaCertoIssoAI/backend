"""
adjudication node — final step in the context agent graph.

runs after evidence gathering is complete.
uses an LLM with structured output to extract claims and adjudicate them
in a single pass, producing a FactCheckResult.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from app.agentic_ai.config import ADJUDICATION_MAX_RETRIES, ADJUDICATION_TIMEOUT
from app.agentic_ai.prompts.adjudication_prompt import build_adjudication_prompt
from app.agentic_ai.state import ContextAgentState
from app.models.factchecking import (
    ClaimVerdict,
    DataSourceResult,
    FactCheckResult,
    LLMAdjudicationOutput,
)

logger = logging.getLogger(__name__)


MAX_REFS_PER_GROUP = 3

# matches consecutive [N] refs, both [1][2][3][4] and [1, 2, 3, 4] styles
_BRACKET_SEQ_RE = re.compile(r"(\[\d+\])(\[\d+\])+")
_COMMA_LIST_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)+)\]")


def _cap_citation_refs(text: str, max_refs: int = MAX_REFS_PER_GROUP) -> str:
    """cap sequences of citation references to max_refs per group.

    handles two patterns the LLM produces:
      [1][2][3][4][5] → [1][2][3]
      [1, 2, 3, 4, 5] → [1][2][3]
    """
    # first: convert comma-list style [1, 2, 3, 4] into individual [1][2][3]
    def _expand_and_cap(m: re.Match) -> str:
        nums = [n.strip() for n in m.group(1).split(",")]
        return "".join(f"[{n}]" for n in nums[:max_refs])

    text = _COMMA_LIST_RE.sub(_expand_and_cap, text)

    # second: cap consecutive [N][N][N]... sequences
    def _cap_consecutive(m: re.Match) -> str:
        refs = re.findall(r"\[\d+\]", m.group(0))
        return "".join(refs[:max_refs])

    text = _BRACKET_SEQ_RE.sub(_cap_consecutive, text)
    return text


def _cap_llm_output_refs(llm_output: LLMAdjudicationOutput) -> None:
    """cap citation refs in all justification and summary fields in-place."""
    for result in llm_output.results:
        for cv in result.claim_verdicts:
            cv.justification = _cap_citation_refs(cv.justification)
    if llm_output.overall_summary:
        llm_output.overall_summary = _cap_citation_refs(llm_output.overall_summary)


def _convert_to_fact_check_result(
    llm_output: LLMAdjudicationOutput,
    formatted_data_sources: str,
) -> FactCheckResult:
    """convert the LLM structured output into a FactCheckResult."""
    data_source_id = str(uuid4())

    all_claim_verdicts: list[ClaimVerdict] = []

    for result in llm_output.results:
        for cv in result.claim_verdicts:
            all_claim_verdicts.append(
                ClaimVerdict(
                    claim_id=cv.claim_id or str(uuid4()),
                    claim_text=cv.claim_text,
                    verdict=cv.verdict,
                    justification=cv.justification,
                    citations_used=cv.citations_used or [],
                )
            )

    data_source_result = DataSourceResult(
        data_source_id=data_source_id,
        source_type="original_text",
        claim_verdicts=all_claim_verdicts,
    )

    return FactCheckResult(
        results=[data_source_result],
        overall_summary=llm_output.overall_summary or None,
    )


def _make_timeout_error_result(attempts: int, timeout: float) -> FactCheckResult:
    """build a fallback FactCheckResult for timeout exhaustion."""
    return FactCheckResult(
        results=[DataSourceResult(
            data_source_id=str(uuid4()),
            source_type="original_text",
            claim_verdicts=[],
        )],
        overall_summary=(
            f"Erro: a adjudicação excedeu o tempo limite de {timeout}s "
            f"após {attempts} tentativa(s)."
        ),
    )


def make_adjudication_node(model: Any):
    """factory that returns the adjudication node function."""

    structured_model = model.with_structured_output(LLMAdjudicationOutput)

    async def adjudication_node(state: ContextAgentState) -> dict:
        system_prompt, user_prompt = build_adjudication_prompt(
            formatted_data_sources=state.get("formatted_data_sources", ""),
            fact_check_results=state.get("fact_check_results", []),
            search_results=state.get("search_results", {}),
            scraped_pages=state.get("scraped_pages", []),
        )

        fc_count = len(state.get("fact_check_results", []))
        sr_count = sum(len(v) for v in state.get("search_results", {}).values())
        sp_count = len(state.get("scraped_pages", []))
        logger.info(
            f"adjudication node: {fc_count} fact_check, {sr_count} search, "
            f"{sp_count} scraped sources in context"
        )

        # build messages once — all retry attempts use the exact same input
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        total_attempts = 1 + ADJUDICATION_MAX_RETRIES
        result: LLMAdjudicationOutput | None = None

        for attempt in range(total_attempts):
            try:
                logger.info(
                    f"adjudication node: invoking LLM (attempt {attempt + 1}/{total_attempts})"
                )
                result = await asyncio.wait_for(
                    structured_model.ainvoke(messages),
                    timeout=ADJUDICATION_TIMEOUT,
                )
                break  # success — exit retry loop
            except asyncio.TimeoutError:
                logger.warning(
                    f"adjudication node: attempt {attempt + 1}/{total_attempts} "
                    f"timed out after {ADJUDICATION_TIMEOUT}s"
                )
                if attempt < ADJUDICATION_MAX_RETRIES:
                    continue
                # all retries exhausted
                error_msg = (
                    f"Adjudication timed out after {total_attempts} attempt(s) "
                    f"({ADJUDICATION_TIMEOUT}s each)"
                )
                logger.error(f"adjudication node: {error_msg}")
                return {
                    "adjudication_result": _make_timeout_error_result(
                        total_attempts, ADJUDICATION_TIMEOUT
                    ),
                    "adjudication_error": error_msg,
                }

        if result is None:
            logger.warning("adjudication node: LLM returned None (schema parse failure)")
            fallback = FactCheckResult(
                results=[DataSourceResult(
                    data_source_id=str(uuid4()),
                    source_type="original_text",
                    claim_verdicts=[],
                )],
                overall_summary="Erro: o modelo nao retornou uma resposta estruturada valida.",
            )
            return {"adjudication_result": fallback}

        for r in result.results:
            for cv in r.claim_verdicts:
                logger.debug(f"[pre-cap] justification: {cv.justification}")
        if result.overall_summary:
            logger.debug(f"[pre-cap] summary: {result.overall_summary}")

        _cap_llm_output_refs(result)

        fact_check_result = _convert_to_fact_check_result(
            result, state.get("formatted_data_sources", "")
        )

        logger.info(
            f"adjudication complete: {len(fact_check_result.results[0].claim_verdicts)} "
            f"claim verdicts produced"
        )

        return {"adjudication_result": fact_check_result}

    return adjudication_node

