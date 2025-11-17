"""
unit tests for async pipeline execution utilities.

tests the fire-and-forget streaming pipeline and its helper functions
to ensure correct structure and error-free execution.
"""

import pytest
from unittest.mock import Mock, patch

from app.ai.async_code import (
    fire_evidence_jobs_for_claim,
    collect_evidence_results,
    fire_and_forget_streaming_pipeline,
)
from app.ai.threads.thread_utils import ThreadPoolManager, OperationType
from app.models import (
    DataSource,
    ClaimExtractionInput,
    ClaimExtractionOutput,
    ExtractedClaim,
    ClaimSource,
    Citation,
    EnrichedClaim,
)


# ===== FIXTURES =====

@pytest.fixture
def mock_thread_pool_manager():
    """create a mock thread pool manager"""
    manager = Mock(spec=ThreadPoolManager)
    manager.submit = Mock()
    manager.wait_next_completed = Mock()
    return manager


@pytest.fixture
def sample_data_source():
    """create a sample data source for testing"""
    return DataSource(
        id="test-source-1",
        source_type="original_text",
        original_text="Sample text for fact checking",
        metadata={}
    )


@pytest.fixture
def sample_claim():
    """create a sample extracted claim"""
    return ExtractedClaim(
        id="claim-1",
        text="Sample claim text",
        source=ClaimSource(source_type="original_text", source_id="test-source-1"),
        entities=["Entity1", "Entity2"],
        llm_comment="This is a testable claim"
    )


@pytest.fixture
def sample_citation():
    """create a sample citation"""
    return Citation(
        url="https://example.com/article",
        title="Test Article",
        publisher="Test Publisher",
        citation_text="Sample citation text",
        source="apify_web_search",
        rating=None,
        date=None
    )


@pytest.fixture
def mock_evidence_gatherer(sample_citation):
    """create a mock evidence gatherer"""
    gatherer = Mock()
    gatherer.source_name = "test_gatherer"
    gatherer.gather_sync = Mock(return_value=[sample_citation])
    return gatherer


# ===== TESTS FOR fire_evidence_jobs_for_claim =====

def test_fire_evidence_jobs_for_claim_structure(
    sample_claim,
    mock_evidence_gatherer,
    mock_thread_pool_manager
):
    """test that fire_evidence_jobs_for_claim submits correct number of jobs"""
    claim_id_to_claim = {}
    evidence_jobs_by_claim = {}

    jobs_submitted = fire_evidence_jobs_for_claim(
        claim=sample_claim,
        evidence_gatherers=[mock_evidence_gatherer],
        manager=mock_thread_pool_manager,
        claim_id_to_claim=claim_id_to_claim,
        evidence_jobs_by_claim=evidence_jobs_by_claim,
    )

    # verify structure
    assert jobs_submitted == 1, "should submit 1 job for 1 gatherer"
    assert sample_claim.id in claim_id_to_claim, "claim should be tracked"
    assert claim_id_to_claim[sample_claim.id] == sample_claim
    assert sample_claim.id in evidence_jobs_by_claim
    assert "test_gatherer" in evidence_jobs_by_claim[sample_claim.id]

    # verify manager.submit was called
    assert mock_thread_pool_manager.submit.call_count == 1
    call_args = mock_thread_pool_manager.submit.call_args
    assert call_args[0][0] == OperationType.LINK_EVIDENCE_RETRIEVER


def test_fire_evidence_jobs_for_claim_multiple_gatherers(
    sample_claim,
    mock_thread_pool_manager
):
    """test firing jobs for multiple evidence gatherers"""
    gatherer1 = Mock()
    gatherer1.source_name = "gatherer_1"
    gatherer1.gather_sync = Mock(return_value=[])

    gatherer2 = Mock()
    gatherer2.source_name = "gatherer_2"
    gatherer2.gather_sync = Mock(return_value=[])

    claim_id_to_claim = {}
    evidence_jobs_by_claim = {}

    jobs_submitted = fire_evidence_jobs_for_claim(
        claim=sample_claim,
        evidence_gatherers=[gatherer1, gatherer2],
        manager=mock_thread_pool_manager,
        claim_id_to_claim=claim_id_to_claim,
        evidence_jobs_by_claim=evidence_jobs_by_claim,
    )

    # verify structure
    assert jobs_submitted == 2, "should submit 2 jobs for 2 gatherers"
    assert len(evidence_jobs_by_claim[sample_claim.id]) == 2
    assert "gatherer_1" in evidence_jobs_by_claim[sample_claim.id]
    assert "gatherer_2" in evidence_jobs_by_claim[sample_claim.id]
    assert mock_thread_pool_manager.submit.call_count == 2


def test_fire_evidence_jobs_for_claim_no_errors(
    sample_claim,
    mock_evidence_gatherer,
    mock_thread_pool_manager
):
    """test that function completes without errors"""
    claim_id_to_claim = {}
    evidence_jobs_by_claim = {}

    # should not raise any exceptions
    try:
        jobs_submitted = fire_evidence_jobs_for_claim(
            claim=sample_claim,
            evidence_gatherers=[mock_evidence_gatherer],
            manager=mock_thread_pool_manager,
            claim_id_to_claim=claim_id_to_claim,
            evidence_jobs_by_claim=evidence_jobs_by_claim,
        )
        assert jobs_submitted >= 0, "function completed without errors"
    except Exception as e:
        pytest.fail(f"function raised unexpected exception: {e}")


# ===== TESTS FOR collect_evidence_results =====

def test_collect_evidence_results_structure(
    sample_claim,
    sample_citation,
    mock_thread_pool_manager
):
    """test that collect_evidence_results returns correct structure"""
    # setup mock to return results
    mock_thread_pool_manager.wait_next_completed.side_effect = [
        ("job-1", (sample_claim.id, [sample_citation])),
    ]

    claim_id_to_claim = {sample_claim.id: sample_claim}

    result = collect_evidence_results(
        manager=mock_thread_pool_manager,
        evidence_jobs_submitted=1,
        claim_id_to_claim=claim_id_to_claim,
    )

    # verify structure
    assert isinstance(result, dict), "result should be a dict"
    assert sample_claim.id in result, "result should contain claim id"
    assert isinstance(result[sample_claim.id], list), "citations should be a list"
    assert len(result[sample_claim.id]) == 1, "should have 1 citation"
    assert result[sample_claim.id][0] == sample_citation


def test_collect_evidence_results_multiple_gatherers(
    sample_claim,
    sample_citation,
    mock_thread_pool_manager
):
    """test collecting results from multiple gatherers for same claim"""
    citation2 = Citation(
        url="https://example2.com",
        title="Test Article 2",
        publisher="Publisher 2",
        citation_text="Citation 2",
        source="google_fact_checking_api",
        rating="Verdadeiro",
        date=None
    )

    # simulate 2 gatherers completing
    mock_thread_pool_manager.wait_next_completed.side_effect = [
        ("job-1", (sample_claim.id, [sample_citation])),
        ("job-2", (sample_claim.id, [citation2])),
    ]

    claim_id_to_claim = {sample_claim.id: sample_claim}

    result = collect_evidence_results(
        manager=mock_thread_pool_manager,
        evidence_jobs_submitted=2,
        claim_id_to_claim=claim_id_to_claim,
    )

    # verify structure
    assert len(result[sample_claim.id]) == 2, "should have 2 citations"
    assert sample_citation in result[sample_claim.id]
    assert citation2 in result[sample_claim.id]


def test_collect_evidence_results_handles_exceptions(
    sample_claim,
    mock_thread_pool_manager
):
    """test that function handles exceptions from gatherers gracefully"""
    # simulate gatherer failure
    mock_thread_pool_manager.wait_next_completed.side_effect = [
        ("job-1", Exception("Gatherer failed")),
    ]

    claim_id_to_claim = {sample_claim.id: sample_claim}

    result = collect_evidence_results(
        manager=mock_thread_pool_manager,
        evidence_jobs_submitted=1,
        claim_id_to_claim=claim_id_to_claim,
    )

    # verify structure - should initialize empty list for claim
    assert sample_claim.id in result
    assert isinstance(result[sample_claim.id], list)
    # exception should be handled, no citations added
    assert len(result[sample_claim.id]) == 0


def test_collect_evidence_results_no_errors(
    sample_claim,
    sample_citation,
    mock_thread_pool_manager
):
    """test that function completes without errors"""
    mock_thread_pool_manager.wait_next_completed.return_value = (
        "job-1",
        (sample_claim.id, [sample_citation])
    )

    claim_id_to_claim = {sample_claim.id: sample_claim}

    try:
        result = collect_evidence_results(
            manager=mock_thread_pool_manager,
            evidence_jobs_submitted=1,
            claim_id_to_claim=claim_id_to_claim,
        )
        assert isinstance(result, dict), "function completed without errors"
    except Exception as e:
        pytest.fail(f"function raised unexpected exception: {e}")


# ===== TESTS FOR fire_and_forget_streaming_pipeline =====

def test_fire_and_forget_pipeline_structure(
    sample_data_source,
    sample_claim,
    sample_citation,
    mock_evidence_gatherer
):
    """test that pipeline returns correct structure"""
    # create mock extract function
    def mock_extract_fn(extraction_input: ClaimExtractionInput):
        return ClaimExtractionOutput(
            data_source=extraction_input.data_source,
            claims=[sample_claim]
        )

    # create mock manager
    with patch('app.ai.async_code.ThreadPoolManager') as MockManager:
        mock_manager = Mock(spec=ThreadPoolManager)
        MockManager.get_instance.return_value = mock_manager

        # simulate claim extraction completion
        mock_manager.wait_next_completed.side_effect = [
            # claim extraction completes
            ("job-1", ClaimExtractionOutput(
                data_source=sample_data_source,
                claims=[sample_claim]
            )),
            # evidence gathering completes
            ("job-2", (sample_claim.id, [sample_citation])),
        ]

        claim_outputs, enriched_claims = fire_and_forget_streaming_pipeline(
            data_sources=[sample_data_source],
            extract_fn=mock_extract_fn,
            evidence_gatherers=[mock_evidence_gatherer],
            manager=mock_manager,
        )

        # verify structure
        assert isinstance(claim_outputs, list), "claim_outputs should be a list"
        assert len(claim_outputs) == 1, "should have 1 claim output"
        assert isinstance(claim_outputs[0], ClaimExtractionOutput)

        assert isinstance(enriched_claims, dict), "enriched_claims should be a dict"
        assert sample_claim.id in enriched_claims
        assert isinstance(enriched_claims[sample_claim.id], EnrichedClaim)
        assert enriched_claims[sample_claim.id].id == sample_claim.id
        assert enriched_claims[sample_claim.id].text == sample_claim.text
        assert isinstance(enriched_claims[sample_claim.id].citations, list)


def test_fire_and_forget_pipeline_multiple_sources(
    sample_citation,
    mock_evidence_gatherer
):
    """test pipeline with multiple data sources"""
    source1 = DataSource(
        id="source-1",
        source_type="original_text",
        original_text="Text 1",
        metadata={}
    )
    source2 = DataSource(
        id="source-2",
        source_type="original_text",
        original_text="Text 2",
        metadata={}
    )

    claim1 = ExtractedClaim(
        id="claim-1",
        text="Claim 1",
        source=ClaimSource(source_type="original_text", source_id="source-1"),
        entities=[],
        llm_comment=None
    )
    claim2 = ExtractedClaim(
        id="claim-2",
        text="Claim 2",
        source=ClaimSource(source_type="original_text", source_id="source-2"),
        entities=[],
        llm_comment=None
    )

    def mock_extract_fn(extraction_input: ClaimExtractionInput):
        if extraction_input.data_source.id == "source-1":
            return ClaimExtractionOutput(
                data_source=extraction_input.data_source,
                claims=[claim1]
            )
        else:
            return ClaimExtractionOutput(
                data_source=extraction_input.data_source,
                claims=[claim2]
            )

    with patch('app.ai.async_code.ThreadPoolManager') as MockManager:
        mock_manager = Mock(spec=ThreadPoolManager)
        MockManager.get_instance.return_value = mock_manager

        # simulate both extractions and evidence gathering completing
        mock_manager.wait_next_completed.side_effect = [
            # claim extractions
            ("job-1", ClaimExtractionOutput(data_source=source1, claims=[claim1])),
            ("job-2", ClaimExtractionOutput(data_source=source2, claims=[claim2])),
            # evidence gathering
            ("job-3", (claim1.id, [sample_citation])),
            ("job-4", (claim2.id, [sample_citation])),
        ]

        claim_outputs, enriched_claims = fire_and_forget_streaming_pipeline(
            data_sources=[source1, source2],
            extract_fn=mock_extract_fn,
            evidence_gatherers=[mock_evidence_gatherer],
            manager=mock_manager,
        )

        # verify structure
        assert len(claim_outputs) == 2, "should have 2 claim outputs"
        assert len(enriched_claims) == 2, "should have 2 enriched claims"
        assert claim1.id in enriched_claims
        assert claim2.id in enriched_claims


def test_fire_and_forget_pipeline_no_errors(
    sample_data_source,
    sample_claim,
    sample_citation,
    mock_evidence_gatherer
):
    """test that pipeline completes without errors"""
    def mock_extract_fn(extraction_input: ClaimExtractionInput):
        return ClaimExtractionOutput(
            data_source=extraction_input.data_source,
            claims=[sample_claim]
        )

    with patch('app.ai.async_code.ThreadPoolManager') as MockManager:
        mock_manager = Mock(spec=ThreadPoolManager)
        MockManager.get_instance.return_value = mock_manager

        mock_manager.wait_next_completed.side_effect = [
            ("job-1", ClaimExtractionOutput(
                data_source=sample_data_source,
                claims=[sample_claim]
            )),
            ("job-2", (sample_claim.id, [sample_citation])),
        ]

        try:
            claim_outputs, enriched_claims = fire_and_forget_streaming_pipeline(
                data_sources=[sample_data_source],
                extract_fn=mock_extract_fn,
                evidence_gatherers=[mock_evidence_gatherer],
                manager=mock_manager,
            )
            assert isinstance(claim_outputs, list), "pipeline completed without errors"
            assert isinstance(enriched_claims, dict), "pipeline completed without errors"
        except Exception as e:
            pytest.fail(f"pipeline raised unexpected exception: {e}")


def test_fire_and_forget_pipeline_empty_claims(
    sample_data_source,
    mock_evidence_gatherer
):
    """test pipeline when no claims are extracted"""
    def mock_extract_fn(extraction_input: ClaimExtractionInput):
        return ClaimExtractionOutput(
            data_source=extraction_input.data_source,
            claims=[]  # no claims
        )

    with patch('app.ai.async_code.ThreadPoolManager') as MockManager:
        mock_manager = Mock(spec=ThreadPoolManager)
        MockManager.get_instance.return_value = mock_manager

        mock_manager.wait_next_completed.side_effect = [
            ("job-1", ClaimExtractionOutput(
                data_source=sample_data_source,
                claims=[]
            )),
            # no evidence jobs since no claims
        ]

        claim_outputs, enriched_claims = fire_and_forget_streaming_pipeline(
            data_sources=[sample_data_source],
            extract_fn=mock_extract_fn,
            evidence_gatherers=[mock_evidence_gatherer],
            manager=mock_manager,
        )

        # verify structure
        assert len(claim_outputs) == 1
        assert len(claim_outputs[0].claims) == 0
        assert len(enriched_claims) == 0, "no enriched claims if no claims extracted"
