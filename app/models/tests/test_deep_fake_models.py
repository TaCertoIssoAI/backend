"""tests for deep-fake detection models in api.py."""

import pytest
from pydantic import ValidationError

from app.models.api import (
    DeepFakeResult,
    DeepFakeVerificationResult,
    Request,
    ContentItem,
    ContentType,
)


# ---- DeepFakeResult ----

def test_deep_fake_result_valid():
    r = DeepFakeResult(
        label="fake",
        score=0.85,
        model_used="frame_sampler(InternVideo2)",
        media_type="video",
        processing_time_ms=1234.5,
    )
    assert r.label == "fake"
    assert r.score == 0.85


def test_deep_fake_result_missing_required_field():
    with pytest.raises(ValidationError):
        DeepFakeResult(label="fake", score=0.5, model_used="m1", media_type="video")


# ---- DeepFakeVerificationResult ----

def test_deep_fake_verification_result_empty():
    v = DeepFakeVerificationResult()
    assert v.results == []


def test_deep_fake_verification_result_with_results():
    v = DeepFakeVerificationResult(results=[
        DeepFakeResult(
            label="fake", score=0.8, model_used="m1",
            media_type="video", processing_time_ms=100,
        ),
        DeepFakeResult(
            label="real", score=0.2, model_used="m1",
            media_type="video", processing_time_ms=100,
        ),
    ])
    assert len(v.results) == 2


# ---- Request backward compat ----

def test_request_without_deep_fake_field():
    req = Request(content=[
        ContentItem(textContent="test claim", type=ContentType.TEXT),
    ])
    assert req.deep_fake_verification_result is None


def test_request_with_deep_fake_field():
    req = Request(
        content=[ContentItem(textContent="test", type=ContentType.TEXT)],
        deep_fake_verification_result=DeepFakeVerificationResult(results=[
            DeepFakeResult(
                label="fake", score=0.9, model_used="m1",
                media_type="video", processing_time_ms=50,
            ),
        ]),
    )
    assert req.deep_fake_verification_result is not None
    assert len(req.deep_fake_verification_result.results) == 1


def test_request_with_hyphenated_alias_json():
    """the JSON key 'deep-fake-verification-result' should parse via alias."""
    payload = {
        "content": [{"textContent": "claim", "type": "text"}],
        "deep-fake-verification-result": {
            "results": [
                {
                    "label": "fake",
                    "score": 0.75,
                    "model_used": "model_x",
                    "media_type": "audio",
                    "processing_time_ms": 200,
                }
            ]
        },
    }
    req = Request.model_validate(payload)
    assert req.deep_fake_verification_result is not None
    assert req.deep_fake_verification_result.results[0].media_type == "audio"


def test_request_with_snake_case_field_name():
    """the Python field name should also work via populate_by_name."""
    payload = {
        "content": [{"textContent": "claim", "type": "text"}],
        "deep_fake_verification_result": {
            "results": [
                {
                    "label": "real",
                    "score": 0.95,
                    "model_used": "model_y",
                    "media_type": "video",
                    "processing_time_ms": 300,
                }
            ]
        },
    }
    req = Request.model_validate(payload)
    assert req.deep_fake_verification_result is not None
    assert req.deep_fake_verification_result.results[0].label == "real"
