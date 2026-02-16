"""
integration test for /text endpoint with text-only content (no links).

this test verifies that:
1. the server is up and running
2. the /text endpoint accepts valid requests
3. text-only content is processed correctly
4. the response matches the expected schema

to run:
    python integration_tests/text_only_test.py

or with pytest:
    pytest integration_tests/text_only_test.py -v -s
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
    """
    print API response in a human-readable format.
    """
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
    """
    check if the server is up and running.

    returns:
        True if server is healthy, False otherwise
    """
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


def test_simple_text_claim():
    """
    test with simple text containing a factual claim.
    """
    print("\n" + "=" * 80)
    print("TEST 1: Simple Text Claim")
    print("=" * 80)

    payload = {
        "content": [
            {
                "textContent": "Neymar est\u00e1 dando tudo de si para tentar tirar o Santos da zona de rebaixamento",
                "type": "text"
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

    print("\nTest passed: Response schema is correct")
    return response_data


def test_multiple_text_claims():
    """
    test with multiple sentences containing different claims.
    """
    print("\n" + "=" * 80)
    print("TEST 2: Multiple Claims in Text")
    print("=" * 80)

    payload = {
        "content": [
            {
                "textContent": "O presidente anunciou um imposto de R$250 por tonelada de carbono. O governo tamb\u00e9m vai investir R$500 bilh\u00f5es em energia renov\u00e1vel nos pr\u00f3ximos 10 anos.",
                "type": "text"
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

    print("\nTest passed: Multiple claims processed successfully")
    return response_data


def test_short_text():
    """
    test with very short text.
    """
    print("\n" + "=" * 80)
    print("TEST 3: Short Text")
    print("=" * 80)

    payload = {
        "content": [
            {
                "textContent": "O Brasil \u00e9 campe\u00e3o mundial.",
                "type": "text"
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

    print("\nTest passed: Short text processed successfully")
    return response_data


def test_long_text():
    """
    test with longer text containing multiple claims.
    """
    print("\n" + "=" * 80)
    print("TEST 4: Long Text with Multiple Claims")
    print("=" * 80)

    payload = {
        "content": [
            {
                "textContent": """
A vacina contra COVID-19 foi desenvolvida em tempo recorde. Estudos mostram que ela \u00e9 95% eficaz na preven\u00e7\u00e3o de casos graves.
O uso de m\u00e1scaras reduziu a transmiss\u00e3o em 70% segundo pesquisas recentes.
Al\u00e9m disso, o distanciamento social foi fundamental para achatar a curva de cont\u00e1gio durante os picos da pandemia.
""".strip(),
                "type": "text"
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

    print("\nTest passed: Long text processed successfully")
    return response_data


def test_opinion_text():
    """
    test with text that is more opinion-based (may not have verifiable claims).
    """
    print("\n" + "=" * 80)
    print("TEST 5: Opinion Text (Less Verifiable)")
    print("=" * 80)

    payload = {
        "content": [
            {
                "textContent": "Eu acho que o time jogou muito bem ontem. Foi uma partida emocionante e divertida de assistir.",
                "type": "text"
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

    print("\nTest passed: Opinion text processed successfully")
    return response_data


def run_all_tests():
    """
    run all integration tests.
    """
    print("\n" + "=" * 80)
    print("TEXT-ONLY INTEGRATION TESTS")
    print("=" * 80)
    print(f"Target: {TEXT_ENDPOINT}")
    print("=" * 80)

    # check server health first
    if not check_server_health():
        print("\nServer is not available. Aborting tests.")
        sys.exit(1)

    # run all tests
    tests = [
        test_simple_text_claim,
        test_multiple_text_claims,
        test_short_text,
        test_long_text,
        test_opinion_text,
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

    # summary
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
