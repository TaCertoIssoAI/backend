"""tests for the adjudication node conversion logic."""

from app.agentic_ai.nodes.adjudication import _convert_to_fact_check_result
from app.models.factchecking import (
    Citation,
    ClaimVerdict,
    DataSourceResult,
    FactCheckResult,
    LLMAdjudicationOutput,
    LLMClaimVerdict,
    LLMDataSourceResult,
)


def _make_llm_output(
    claim_verdicts: list[LLMClaimVerdict] | None = None,
    overall_summary: str = "summary",
    num_results: int = 1,
) -> LLMAdjudicationOutput:
    """helper to build an LLMAdjudicationOutput."""
    if claim_verdicts is None:
        claim_verdicts = [
            LLMClaimVerdict(
                claim_id=None,
                claim_text="test claim",
                verdict="Falso",
                justification="contradicted by [1]",
                citations_used=[],
            )
        ]
    results = [
        LLMDataSourceResult(data_source_id=None, claim_verdicts=claim_verdicts)
        for _ in range(num_results)
    ]
    return LLMAdjudicationOutput(results=results, overall_summary=overall_summary)


def test_returns_fact_check_result():
    llm_output = _make_llm_output()
    result = _convert_to_fact_check_result(llm_output, "original text")
    assert isinstance(result, FactCheckResult)


def test_single_claim_verdict():
    llm_output = _make_llm_output()
    result = _convert_to_fact_check_result(llm_output, "original text")

    assert len(result.results) == 1
    ds = result.results[0]
    assert ds.source_type == "original_text"
    assert len(ds.claim_verdicts) == 1
    assert ds.claim_verdicts[0].claim_text == "test claim"
    assert ds.claim_verdicts[0].verdict == "Falso"
    assert ds.claim_verdicts[0].justification == "contradicted by [1]"


def test_multiple_claims_flattened():
    """claims from multiple LLMDataSourceResults are flattened into one DataSourceResult."""
    cv1 = LLMClaimVerdict(
        claim_text="claim A", verdict="Verdadeiro",
        justification="confirmed by [1]",
    )
    cv2 = LLMClaimVerdict(
        claim_text="claim B", verdict="Falso",
        justification="denied by [2]",
    )
    llm_output = LLMAdjudicationOutput(
        results=[
            LLMDataSourceResult(data_source_id=None, claim_verdicts=[cv1]),
            LLMDataSourceResult(data_source_id=None, claim_verdicts=[cv2]),
        ],
        overall_summary="two claims",
    )
    result = _convert_to_fact_check_result(llm_output, "text")

    assert len(result.results) == 1
    verdicts = result.results[0].claim_verdicts
    assert len(verdicts) == 2
    assert verdicts[0].claim_text == "claim A"
    assert verdicts[1].claim_text == "claim B"


def test_claim_id_generated_when_missing():
    llm_output = _make_llm_output(
        claim_verdicts=[
            LLMClaimVerdict(
                claim_id=None, claim_text="no id", verdict="Falso",
                justification="reason",
            )
        ]
    )
    result = _convert_to_fact_check_result(llm_output, "text")
    claim_id = result.results[0].claim_verdicts[0].claim_id
    assert claim_id is not None
    assert len(claim_id) > 0


def test_claim_id_preserved_when_present():
    llm_output = _make_llm_output(
        claim_verdicts=[
            LLMClaimVerdict(
                claim_id="custom-id", claim_text="with id", verdict="Verdadeiro",
                justification="reason",
            )
        ]
    )
    result = _convert_to_fact_check_result(llm_output, "text")
    assert result.results[0].claim_verdicts[0].claim_id == "custom-id"


def test_overall_summary_preserved():
    llm_output = _make_llm_output(overall_summary="the summary")
    result = _convert_to_fact_check_result(llm_output, "text")
    assert result.overall_summary == "the summary"


def test_empty_overall_summary_becomes_none():
    llm_output = _make_llm_output(overall_summary="")
    result = _convert_to_fact_check_result(llm_output, "text")
    assert result.overall_summary is None


def test_citations_preserved():
    citation = Citation(
        url="https://example.com",
        title="Example",
        publisher="Pub",
        citation_text="relevant excerpt",
        source="google_web_search",
    )
    llm_output = _make_llm_output(
        claim_verdicts=[
            LLMClaimVerdict(
                claim_text="cited claim", verdict="Falso",
                justification="see [1]",
                citations_used=[citation],
            )
        ]
    )
    result = _convert_to_fact_check_result(llm_output, "text")
    cv = result.results[0].claim_verdicts[0]
    assert len(cv.citations_used) == 1
    assert cv.citations_used[0].url == "https://example.com"
    assert cv.citations_used[0].source == "google_web_search"


def test_empty_citations_default():
    llm_output = _make_llm_output(
        claim_verdicts=[
            LLMClaimVerdict(
                claim_text="no cites", verdict="Fontes insuficientes para verificar",
                justification="no sources",
            )
        ]
    )
    result = _convert_to_fact_check_result(llm_output, "text")
    assert result.results[0].claim_verdicts[0].citations_used == []


def test_no_results_produces_empty_verdicts():
    llm_output = LLMAdjudicationOutput(results=[], overall_summary="nothing")
    result = _convert_to_fact_check_result(llm_output, "text")
    assert len(result.results) == 1
    assert result.results[0].claim_verdicts == []


def test_data_source_id_is_uuid():
    """each call generates a unique data_source_id."""
    llm_output = _make_llm_output()
    r1 = _convert_to_fact_check_result(llm_output, "text")
    r2 = _convert_to_fact_check_result(llm_output, "text")
    assert r1.results[0].data_source_id != r2.results[0].data_source_id


def test_all_verdict_types():
    """all four verdict types pass through correctly."""
    verdicts = [
        "Verdadeiro", "Falso", "Fora de Contexto",
        "Fontes insuficientes para verificar",
    ]
    cvs = [
        LLMClaimVerdict(
            claim_text=f"claim {v}", verdict=v, justification="reason"
        )
        for v in verdicts
    ]
    llm_output = _make_llm_output(claim_verdicts=cvs)
    result = _convert_to_fact_check_result(llm_output, "text")
    result_verdicts = [cv.verdict for cv in result.results[0].claim_verdicts]
    assert result_verdicts == verdicts
