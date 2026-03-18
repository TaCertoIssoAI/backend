"""
integration test for /text endpoint with deep-fake verification results.

this test verifies that:
1. the server is up and running
2. the /text endpoint accepts requests with deep-fake results
3. requests without deep-fake results still work (backward compat)
4. the response matches the expected schema

to run:
    python integration_tests/deep_fake_test.py

or with pytest:
    pytest integration_tests/deep_fake_test.py -v -s
"""

import requests
import json
import sys
import time


# configuration
BASE_URL = "http://localhost:8000"
TEXT_ENDPOINT = f"{BASE_URL}/text"
HEALTH_ENDPOINT = f"{BASE_URL}/health"


def pretty_print_response(response_data: dict):
    """print API response in a human-readable format."""
    print("\n" + "=" * 80)
    print("RESPONSE SUMMARY")
    print("=" * 80)

    if "message_id" in response_data:
        print(f"\nMessage ID: {response_data['message_id']}")

    if "rationale" in response_data:
        print(f"\nRationale:")
        print(f"   {response_data['rationale']}")

    if "responseWithoutLinks" in response_data:
        print(f"\nResponse (without links):")
        print(f"   {response_data['responseWithoutLinks']}")

    print("\n" + "=" * 80)


def check_server_health() -> bool:
    """check if the server is up and running."""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        if response.status_code == 200:
            print("Server is up and running")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"Server returned status code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"Could not connect to server at {BASE_URL}")
        print("   Make sure the server is running: uvicorn app.main:app --reload --port 8000")
        return False
    except requests.exceptions.Timeout:
        print("Server health check timed out")
        return False
    except Exception as e:
        print(f"Unexpected error checking server health: {e}")
        return False


def _assert_response_schema(response_data: dict):
    """validate that the response matches the AnalysisResponse schema."""
    assert "message_id" in response_data, "Response missing 'message_id'"
    assert "rationale" in response_data, "Response missing 'rationale'"
    assert "responseWithoutLinks" in response_data, "Response missing 'responseWithoutLinks'"
    assert len(response_data["rationale"]) > 0, "Rationale is empty"
    assert len(response_data["responseWithoutLinks"]) > 0, "responseWithoutLinks is empty"


def test_deep_fake_high_score():
    """test with video content + deep-fake results where all models have fake > 0.7."""
    print("\n" + "=" * 80)
    print("TEST 1: Deep Fake High Score (all models fake > 0.7)")
    print("=" * 80)

    payload = {
        "content": [
            {
                "textContent": "Neste video, o presidente Lula afirma que o salario minimo no Brasil vai subir para R$5.000 em 2025",
                "type": "video",
            }
        ],
        "deep-fake-verification-result": {
            "results": [
                {
                    "label": "fake",
                    "score": 0.8392,
                    "model_used": "frame_sampler(InternVideo2)",
                    "media_type": "video",
                    "processing_time_ms": 1500,
                },
                {
                    "label": "real",
                    "score": 0.1608,
                    "model_used": "frame_sampler(InternVideo2)",
                    "media_type": "video",
                    "processing_time_ms": 1500,
                },
                {
                    "label": "fake",
                    "score": 0.7821,
                    "model_used": "VoiceGen (Dual-RawNet2)",
                    "media_type": "audio",
                    "processing_time_ms": 800,
                },
                {
                    "label": "real",
                    "score": 0.2179,
                    "model_used": "VoiceGen (Dual-RawNet2)",
                    "media_type": "audio",
                    "processing_time_ms": 800,
                },
            ]
        },
    }

    print(f"\nRequest payload:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    start_time = time.time()
    response = requests.post(TEXT_ENDPOINT, json=payload, timeout=120)
    elapsed_time = time.time() - start_time

    print(f"\nRequest time: {elapsed_time:.2f} seconds ({elapsed_time * 1000:.0f} ms)")
    print(f"Response status: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    response_data = response.json()
    pretty_print_response(response_data)
    _assert_response_schema(response_data)

    print("\nTest passed: Deep fake high score processed successfully")
    return response_data


def test_deep_fake_low_score():
    """test with video content + deep-fake results where fake score < 0.6."""
    print("\n" + "=" * 80)
    print("TEST 2: Deep Fake Low Score (fake < 0.6)")
    print("=" * 80)

    payload = {
        "content": [
            {
                "textContent": "Video mostra o ministro da Fazenda Fernando Haddad anunciando que o imposto de renda sera isento ate R$5.000 a partir de 2026",
                "type": "video",
            }
        ],
        "deep-fake-verification-result": {
            "results": [
                {
                    "label": "fake",
                    "score": 0.15,
                    "model_used": "frame_sampler(InternVideo2)",
                    "media_type": "video",
                    "processing_time_ms": 1200,
                },
                {
                    "label": "real",
                    "score": 0.85,
                    "model_used": "frame_sampler(InternVideo2)",
                    "media_type": "video",
                    "processing_time_ms": 1200,
                },
            ]
        },
    }

    print(f"\nRequest payload:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    start_time = time.time()
    response = requests.post(TEXT_ENDPOINT, json=payload, timeout=120)
    elapsed_time = time.time() - start_time

    print(f"\nRequest time: {elapsed_time:.2f} seconds ({elapsed_time * 1000:.0f} ms)")
    print(f"Response status: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    response_data = response.json()
    pretty_print_response(response_data)
    _assert_response_schema(response_data)

    print("\nTest passed: Deep fake low score processed successfully")
    return response_data


def test_no_deep_fake_backward_compat():
    """test that requests without deep-fake field still work."""
    print("\n" + "=" * 80)
    print("TEST 3: No Deep Fake Field (Backward Compat)")
    print("=" * 80)

    payload = {
        "content": [
            {
                "textContent": "O governo anunciou novas medidas economicas",
                "type": "text",
            }
        ]
    }

    print(f"\nRequest payload:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    start_time = time.time()
    response = requests.post(TEXT_ENDPOINT, json=payload, timeout=120)
    elapsed_time = time.time() - start_time

    print(f"\nRequest time: {elapsed_time:.2f} seconds ({elapsed_time * 1000:.0f} ms)")
    print(f"Response status: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    response_data = response.json()
    pretty_print_response(response_data)
    _assert_response_schema(response_data)

    print("\nTest passed: Backward compat works")
    return response_data


def run_all_tests():
    """run all integration tests."""
    print("\n" + "=" * 80)
    print("DEEP-FAKE INTEGRATION TESTS")
    print("=" * 80)
    print(f"Target: {TEXT_ENDPOINT}")
    print("=" * 80)

    if not check_server_health():
        print("\nServer is not available. Aborting tests.")
        sys.exit(1)

    tests = [
        test_deep_fake_high_score,
        test_deep_fake_low_score,
        test_no_deep_fake_backward_compat,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\nTest failed: {e}")
            failed += 1
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            failed += 1

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    print("=" * 80)

    if failed > 0:
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
