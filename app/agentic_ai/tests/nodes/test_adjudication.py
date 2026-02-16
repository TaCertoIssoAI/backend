"""tests for the adjudication node conversion logic."""

from app.agentic_ai.nodes.adjudication import (
    _cap_citation_refs,
    _cap_llm_output_refs,
    _convert_to_fact_check_result,
)
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


# --- _cap_citation_refs ---


def test_cap_consecutive_brackets():
    assert _cap_citation_refs("[1][2][3][4][5]") == "[1][2][3]"


def test_cap_comma_list():
    assert _cap_citation_refs("[1, 2, 3, 4, 5]") == "[1][2][3]"


def test_cap_long_comma_list():
    assert _cap_citation_refs("[4, 6, 7, 8, 11, 13, 17, 36, 39, 40]") == "[4][6][7]"


def test_preserves_three_or_fewer_brackets():
    assert _cap_citation_refs("[1][2]") == "[1][2]"
    assert _cap_citation_refs("[1][2][3]") == "[1][2][3]"


def test_preserves_single_ref():
    assert _cap_citation_refs("text [1] more text") == "text [1] more text"


def test_caps_multiple_groups_independently():
    text = "point A [1][2][3][4][5] and point B [6][7][8][9]"
    result = _cap_citation_refs(text)
    assert result == "point A [1][2][3] and point B [6][7][8]"


def test_mixed_styles_in_same_text():
    text = "first [1, 2, 3, 4] then [5][6][7][8][9]"
    result = _cap_citation_refs(text)
    assert result == "first [1][2][3] then [5][6][7]"


def test_no_refs_unchanged():
    text = "no references here"
    assert _cap_citation_refs(text) == text


def test_cap_llm_output_refs_modifies_in_place():
    llm_output = _make_llm_output(
        claim_verdicts=[
            LLMClaimVerdict(
                claim_text="claim", verdict="Falso",
                justification="sources [1][2][3][4][5] confirm",
            )
        ],
        overall_summary="summary [1][2][3][4]",
    )
    _cap_llm_output_refs(llm_output)
    assert llm_output.results[0].claim_verdicts[0].justification == "sources [1][2][3] confirm"
    assert llm_output.overall_summary == "summary [1][2][3]"


def test_cap_exactly_four_consecutive():
    assert _cap_citation_refs("[1][2][3][4]") == "[1][2][3]"


def test_cap_two_consecutive_unchanged():
    """two consecutive refs are below the cap — should pass through untouched."""
    assert _cap_citation_refs("[5][9]") == "[5][9]"


def test_cap_high_number_refs():
    assert _cap_citation_refs("[10][20][30][40]") == "[10][20][30]"


def test_real_world_long_sequence():
    """exact sequence from production output."""
    text = "[4][6][7][8][11][13][17][36][39][40]"
    assert _cap_citation_refs(text) == "[4][6][7]"


def test_refs_mid_sentence_with_surrounding_text():
    text = "A alegacao e confirmada [1][2][3][4][5] por diversas fontes."
    assert _cap_citation_refs(text) == "A alegacao e confirmada [1][2][3] por diversas fontes."


def test_refs_at_end_of_sentence():
    text = "A alegacao e falsa [1][2][3][4]."
    assert _cap_citation_refs(text) == "A alegacao e falsa [1][2][3]."


def test_single_refs_separated_by_text_unchanged():
    """single refs with text between them are not consecutive — should be unchanged."""
    text = "A [1] e B [2] e C [3] confirmam."
    assert _cap_citation_refs(text) == text


def test_comma_list_with_two_items_normalized():
    """comma list with 2 items converts to bracket style but doesn't truncate."""
    assert _cap_citation_refs("[1, 2]") == "[1][2]"


def test_comma_list_with_exactly_three():
    assert _cap_citation_refs("[1, 2, 3]") == "[1][2][3]"


def test_comma_list_with_spaces_around_numbers():
    assert _cap_citation_refs("[ 1 , 2 , 3 , 4 , 5 ]") == "[ 1 , 2 , 3 , 4 , 5 ]"  # spaces inside brackets don't match \d+


def test_comma_list_with_spaces_around_commas():
    assert _cap_citation_refs("[1 , 2 , 3 , 4]") == "[1][2][3]"


def test_group_of_two_then_group_of_four():
    text = "primeiro [1][2] depois [3][4][5][6][7]"
    assert _cap_citation_refs(text) == "primeiro [1][2] depois [3][4][5]"


def test_empty_string():
    assert _cap_citation_refs("") == ""


def test_non_numeric_brackets_unchanged():
    """brackets with non-numeric content should not be affected."""
    assert _cap_citation_refs("[item][other]") == "[item][other]"


def test_comma_list_then_consecutive_refs():
    """comma list converted first, then consecutive check applied on result."""
    text = "A [1, 2, 3, 4] e B [5][6][7][8]"
    assert _cap_citation_refs(text) == "A [1][2][3] e B [5][6][7]"


def test_cap_llm_output_refs_multiple_claims():
    """capping applies to every claim verdict independently."""
    llm_output = _make_llm_output(
        claim_verdicts=[
            LLMClaimVerdict(
                claim_text="claim A", verdict="Falso",
                justification="reason A [1][2][3][4][5]",
            ),
            LLMClaimVerdict(
                claim_text="claim B", verdict="Verdadeiro",
                justification="reason B [6, 7, 8, 9]",
            ),
        ],
        overall_summary="both claims checked [1][2][3][4]",
    )
    _cap_llm_output_refs(llm_output)
    cvs = llm_output.results[0].claim_verdicts
    assert cvs[0].justification == "reason A [1][2][3]"
    assert cvs[1].justification == "reason B [6][7][8]"
    assert llm_output.overall_summary == "both claims checked [1][2][3]"
