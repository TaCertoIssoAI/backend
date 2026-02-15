"""
adjudication node — final step in the context agent graph.

runs after evidence gathering is complete.
uses an LLM with structured output to extract claims and adjudicate them
in a single pass, producing a FactCheckResult.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from app.agentic_ai.prompts.adjudication_prompt import build_adjudication_prompt
from app.agentic_ai.state import ContextAgentState
from app.models.factchecking import (
    Citation,
    ClaimVerdict,
    DataSourceResult,
    FactCheckResult,
    LLMAdjudicationOutput,
)

logger = logging.getLogger(__name__)


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

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        logger.info("adjudication node: invoking LLM with structured output")
        result: LLMAdjudicationOutput = await structured_model.ainvoke(messages)

        fact_check_result = _convert_to_fact_check_result(
            result, state.get("formatted_data_sources", "")
        )

        logger.info(
            f"adjudication complete: {len(fact_check_result.results[0].claim_verdicts)} "
            f"claim verdicts produced"
        )

        return {"adjudication_result": fact_check_result}

    return adjudication_node
