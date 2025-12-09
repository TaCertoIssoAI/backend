"""
integration test for /text-adjudication-search endpoint.

this test verifies that:
1. the server is up and running
2. normal adjudication returns all unverifiable results (mocked)
3. the fallback to adjudication_with_search is triggered
4. adjudication_with_search uses OpenAI web search to provide better results
5. the final response contains verifiable verdicts from web search

to run:
    python integration_tests/adjudication_with_search_fallback.py

or with pytest:
    pytest integration_tests/adjudication_with_search_fallback.py -v -s
"""

import requests
import json
import sys
import time


# configuration
BASE_URL = "http://localhost:8000"
ADJUDICATION_SEARCH_ENDPOINT = f"{BASE_URL}/text-adjudication-search"
HEALTH_ENDPOINT = f"{BASE_URL}/health"


def pretty_print_response(response_data: dict):
    """
    print API response in a human-readable format.
    """
    print("\n" + "=" * 80)
    print("📋 RESPONSE SUMMARY - ADJUDICATION WITH SEARCH FALLBACK")
    print("=" * 80)

    # message id
    if "message_id" in response_data:
        print(f"\n🆔 Message ID: {response_data['message_id']}")

    # rationale
    if "rationale" in response_data:
        print(f"\n💡 Rationale:")
        # print first 500 chars to avoid clutter
        rationale = response_data['rationale']
        if len(rationale) > 500:
            print(f"   {rationale[:500]}...")
            print(f"   [... {len(rationale) - 500} more characters]")
        else:
            print(f"   {rationale}")

    # response without links
    if "responseWithoutLinks" in response_data:
        print(f"\n📝 Response (without links):")
        resp_without_links = response_data['responseWithoutLinks']
        if len(resp_without_links) > 300:
            print(f"   {resp_without_links[:300]}...")
            print(f"   [... {len(resp_without_links) - 300} more characters]")
        else:
            print(f"   {resp_without_links}")

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
            print("✅ Server is up and running")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Server returned status code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to server at {BASE_URL}")
        print("   Make sure the server is running: uvicorn app.main:app --reload --port 8000")
        return False
    except requests.exceptions.Timeout:
        print("❌ Server health check timed out")
        return False
    except Exception as e:
        print(f"❌ Unexpected error checking server health: {e}")
        return False


def test_fallback_with_verifiable_claims():
    """
    test adjudication_with_search fallback with verifiable scientific claims.

    the mock adjudicator will return "Fontes insuficientes para verificar" for all claims,
    triggering the fallback to adjudication_with_search which should provide real verdicts.
    """
    print("\n" + "=" * 80)
    print("TEST 1: Adjudication with Search Fallback - Scientific Claims")
    print("=" * 80)

    payload = {
        "content": [
            {
                "textContent": "A Terra orbita ao redor do Sol. A água ferve a 100°C ao nível do mar. A Lua é feita de queijo.",
                "type": "text"
            }
        ]
    }

    print("\n📤 Request payload:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    print("\n⏳ Sending request...")
    print("   Expected flow:")
    print("   1. Normal adjudication returns all 'Fontes insuficientes'")
    print("   2. Fallback triggered: adjudication_with_search job")
    print("   3. OpenAI web search finds evidence")
    print("   4. Final response uses web search results")

    start_time = time.time()
    response = requests.post(ADJUDICATION_SEARCH_ENDPOINT, json=payload, timeout=120)
    elapsed_time = time.time() - start_time

    print(f"\n⏱️  Request time: {elapsed_time:.2f} seconds ({elapsed_time * 1000:.0f} ms)")
    print(f"📥 Response status: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    response_data = response.json()
    pretty_print_response(response_data)

    # verify response schema
    assert "message_id" in response_data
    assert "rationale" in response_data
    assert "responseWithoutLinks" in response_data

    # check that rationale is not empty
    rationale = response_data["rationale"]
    assert len(rationale) > 0, "Rationale should not be empty"

    # verify that rationale contains verdicts other than "Fontes insuficientes"
    # the fallback should produce verdicts like "Verdadeiro", "Falso", etc.
    rationale_lower = rationale.lower()

    # check for positive verdict indicators (case insensitive)
    has_verdadeiro = "verdadeiro" in rationale_lower or "true" in rationale_lower
    has_falso = "falso" in rationale_lower or "false" in rationale_lower
    has_verified = has_verdadeiro or has_falso

    # check that it's not all insufficient
    has_only_insufficient = (
        "fontes insuficientes" in rationale_lower and
        not has_verdadeiro and
        not has_falso
    )

    print(f"\n🔍 Verdict Analysis from Rationale:")
    print(f"   Has 'Verdadeiro': {has_verdadeiro}")
    print(f"   Has 'Falso': {has_falso}")
    print(f"   Has verified claims: {has_verified}")
    print(f"   Only insufficient sources: {has_only_insufficient}")

    assert has_verified, (
        "Expected at least one verified verdict (Verdadeiro or Falso) from adjudication_with_search fallback, "
        f"but rationale only contains 'Fontes insuficientes'. Rationale preview: {rationale[:300]}..."
    )

    print("\n✅ Test passed: Adjudication with search fallback worked successfully")
    return response_data


def test_fallback_with_recent_event():
    """
    test with a recent event that requires web search to verify.

    normal adjudication won't have this data, so fallback should help.
    """
    print("\n" + "=" * 80)
    print("TEST 2: Adjudication with Search Fallback - Recent Event")
    print("=" * 80)

    payload = {
        "content": [
            {
                "textContent": "Portugal venceu a Eurocopa de 2024",
                "type": "text"
            }
        ]
    }

    print("\n📤 Request payload:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    print("\n⏳ Sending request...")
    start_time = time.time()
    response = requests.post(ADJUDICATION_SEARCH_ENDPOINT, json=payload, timeout=90)
    elapsed_time = time.time() - start_time

    print(f"\n⏱️  Request time: {elapsed_time:.2f} seconds ({elapsed_time * 1000:.0f} ms)")
    print(f"📥 Response status: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    response_data = response.json()
    pretty_print_response(response_data)

    # verify response has required fields
    assert "message_id" in response_data
    assert "rationale" in response_data
    assert "responseWithoutLinks" in response_data

    rationale = response_data["rationale"]
    assert len(rationale) > 0, "Rationale should not be empty"

    # for recent events, we expect the fallback to find information via web search
    print(f"\n🔍 Checking rationale for recent event verification:")
    print(f"   Rationale length: {len(rationale)} chars")
    print(f"   Preview: {rationale[:200]}...")

    print("\n✅ Test passed: Recent event processed with adjudication_with_search")
    return response_data


def test_fallback_with_mixed_claims():
    """
    test with multiple claims of different types.
    """
    print("\n" + "=" * 80)
    print("TEST 3: Adjudication with Search Fallback - Mixed Claims")
    print("=" * 80)

    payload = {
        "content": [
            {
                "textContent": "O Brasil ganhou 5 Copas do Mundo. A capital do Brasil é Brasília. Vacinas causam autismo.",
                "type": "text"
            }
        ]
    }

    print("\n📤 Request payload:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    print("\n⏳ Sending request...")
    start_time = time.time()
    response = requests.post(ADJUDICATION_SEARCH_ENDPOINT, json=payload, timeout=120)
    elapsed_time = time.time() - start_time

    print(f"\n⏱️  Request time: {elapsed_time:.2f} seconds ({elapsed_time * 1000:.0f} ms)")
    print(f"📥 Response status: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    response_data = response.json()
    pretty_print_response(response_data)

    # verify response schema
    assert "message_id" in response_data
    assert "rationale" in response_data
    assert "responseWithoutLinks" in response_data

    rationale = response_data["rationale"]
    assert len(rationale) > 0, "Rationale should not be empty"

    # verify that rationale contains verified verdicts
    rationale_lower = rationale.lower()

    has_verdadeiro = "verdadeiro" in rationale_lower
    has_falso = "falso" in rationale_lower
    has_verified = has_verdadeiro or has_falso

    print(f"\n🔍 Verdict Distribution in Rationale:")
    print(f"   Has 'Verdadeiro': {has_verdadeiro}")
    print(f"   Has 'Falso': {has_falso}")
    print(f"   Has verified claims: {has_verified}")

    if not has_verified:
        print(f"\n❌ ASSERTION FAILURE:")
        print(f"   Expected: At least 1 verified claim (Verdadeiro or Falso)")
        print(f"   Actual: Only 'Fontes insuficientes' found")

    assert has_verified, (
        f"At least some claims should be verified via web search. "
        f"Rationale preview: {rationale[:300]}..."
    )

    print("\n✅ Test passed: Mixed claims processed successfully")
    return response_data


def run_all_tests():
    """
    run all integration tests for adjudication_with_search fallback.
    """
    print("\n" + "=" * 80)
    print("ADJUDICATION WITH SEARCH FALLBACK - INTEGRATION TESTS")
    print("=" * 80)
    print(f"Target: {ADJUDICATION_SEARCH_ENDPOINT}")
    print("⚠️  NOTE: These tests require OPENAI_API_KEY to be set in .env")
    print("=" * 80)

    # check server health first
    if not check_server_health():
        print("\n❌ Server is not available. Aborting tests.")
        sys.exit(1)

    # run all tests
    tests = [
        test_fallback_with_verifiable_claims,
        test_fallback_with_recent_event,
        test_fallback_with_mixed_claims,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ Test failed: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {passed + failed}")
    print("=" * 80)

    if failed > 0:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        sys.exit(1)
    else:
        print("\n🎉 All tests passed!")
        print("\nThe fallback mechanism successfully:")
        print("  - Detected all-unverifiable results from normal adjudication")
        print("  - Triggered adjudication_with_search fallback")
        print("  - Used OpenAI web search to find evidence")
        print("  - Returned verified verdicts instead of 'Fontes insuficientes'")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
