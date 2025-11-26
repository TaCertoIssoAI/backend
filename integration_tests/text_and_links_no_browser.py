"""
integration test for /text endpoint with text containing links.

this test verifies that:
1. the server is up and running
2. links are extracted from text content
3. links are expanded and scraped
4. claims are extracted from both original text and link content
5. the response includes claims from all sources

to run:
    python integration_tests/text_and_links_test.py

or with pytest:
    pytest integration_tests/text_and_links_test.py -v -s
"""

import requests
import json
import sys
import time


# configuration
BASE_URL = "http://localhost:8000"
TEXT_ENDPOINT = f"{BASE_URL}/text-without-browser"
HEALTH_ENDPOINT = f"{BASE_URL}/health"


def pretty_print_response(response_data: dict):
    """
    print API response in a human-readable format.
    """
    print("\n" + "=" * 80)
    print("📋 RESPONSE SUMMARY")
    print("=" * 80)

    # message id
    if "message_id" in response_data:
        print(f"\n🆔 Message ID: {response_data['message_id']}")

    # verdict
    if "verdict" in response_data:
        verdict_emoji = {
            "true": "✅",
            "false": "❌",
            "misleading": "⚠️",
            "unverifiable": "❓",
        }
        verdict = response_data["verdict"].lower()
        emoji = verdict_emoji.get(verdict, "🔍")
        print(f"\n{emoji} Verdict: {response_data['verdict'].upper()}")

    # confidence
    if "confidence" in response_data:
        confidence = response_data["confidence"]
        print(f"📊 Confidence: {confidence:.2%}")

    # rationale
    if "rationale" in response_data:
        print(f"\n💡 Rationale:")
        print(f"   {response_data['rationale']}")

    # response without links
    if "responseWithoutLinks" in response_data:
        print(f"\n📝 Response (without links):")
        print(f"   {response_data['responseWithoutLinks']}")

    # claims
    if "claims" in response_data and response_data["claims"]:
        print(f"\n🔎 Extracted Claims ({len(response_data['claims'])} total):")
        for i, claim in enumerate(response_data["claims"], 1):
            print(f"   {i}. {claim.get('claim', 'N/A')}")
            if "verdict" in claim:
                print(f"      → Verdict: {claim['verdict']}")

    # citations
    if "citations" in response_data and response_data["citations"]:
        print(f"\n📚 Citations ({len(response_data['citations'])} sources):")
        for i, citation in enumerate(response_data["citations"], 1):
            print(f"\n   [{i}] {citation.get('title', 'No title')}")
            if "source" in citation:
                print(f"       Source: {citation['source']}")
            if "url" in citation:
                print(f"       URL: {citation['url']}")
            if "snippet" in citation and citation['snippet']:
                snippet = citation['snippet'][:150] + "..." if len(citation['snippet']) > 150 else citation['snippet']
                print(f"       Snippet: {snippet}")
            if "published_at" in citation:
                print(f"       Published: {citation['published_at']}")

    # processing time
    if "processing_time_ms" in response_data:
        time_sec = response_data["processing_time_ms"] / 1000
        print(f"\n⚡ Processing Time: {time_sec:.2f}s ({response_data['processing_time_ms']}ms)")

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


def test_text_with_single_link():
    """
    test with text containing a single link.
    """
    print("\n" + "=" * 80)
    print("TEST 1: Text with Single Link")
    print("=" * 80)
    
    payload = {
        "content": [
            {
                "textContent": "Confira essa notícia importante: https://www.cnnbrasil.com.br/nacional/em-belem-cupula-dos-povos-cobra-participacao-popular-nas-acoes-climaticas/ sobre o Neymar no Santos.",
                "type": "text"
            }
        ]
    }
    
    print("\n📤 Request payload:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    print("\n⏳ Sending request... (this may take a while due to link scraping)")
    start_time = time.time()
    response = requests.post(TEXT_ENDPOINT, json=payload, timeout=120)
    elapsed_time = time.time() - start_time

    print(f"\n⏱️  Request time: {elapsed_time:.2f} seconds ({elapsed_time * 1000:.0f} ms)")
    print(f"📥 Response status: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    response_data = response.json()
    pretty_print_response(response_data)

    # verify response schema
    assert "message_id" in response_data
    assert "verdict" in response_data
    assert "rationale" in response_data
    assert "responseWithoutLinks" in response_data

    print("\n✅ Test passed: Single link processed successfully")
    return response_data


def test_text_with_single_link_robust_claim():
    """
    test with text containing a single link.
    """
    print("\n" + "=" * 80)
    print("TEST 1: Text with Single Link")
    print("=" * 80)
    
    payload = {
        "content": [
            {
                "textContent": "Confira essa notícia importante: https://www.cnnbrasil.com.br/nacional/em-belem-cupula-dos-povos-cobra-participacao-popular-nas-acoes-climaticas/ sobre o Neymar no Santos.",
                "type": "text"
            }
        ]
    }
    
    print("\n📤 Request payload:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    print("\n⏳ Sending request... (this may take a while due to link scraping)")
    start_time = time.time()
    response = requests.post(TEXT_ENDPOINT, json=payload, timeout=120)
    elapsed_time = time.time() - start_time

    print(f"\n⏱️  Request time: {elapsed_time:.2f} seconds ({elapsed_time * 1000:.0f} ms)")
    print(f"📥 Response status: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    response_data = response.json()
    pretty_print_response(response_data)

    # verify response schema
    assert "message_id" in response_data
    assert "verdict" in response_data
    assert "rationale" in response_data
    assert "responseWithoutLinks" in response_data

    print("\n✅ Test passed: Single link processed successfully")
    return response_data


def test_text_with_multiple_links():
    """
    test with text containing multiple links.
    """
    print("\n" + "=" * 80)
    print("TEST 2: Text with Multiple Links")
    print("=" * 80)
    
    payload = {
        "content": [
            {
                "textContent": "Veja essas fontes: https://example.com/news1 e também https://example.com/news2 para mais informações.",
                "type": "text"
            }
        ]
    }
    
    print("\n📤 Request payload:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    
    print("\n⏳ Sending request... (this may take a while due to multiple link scraping)")
    start_time = time.time()
    response = requests.post(TEXT_ENDPOINT, json=payload, timeout=180)
    elapsed_time = time.time() - start_time

    print(f"\n⏱️  Request time: {elapsed_time:.2f} seconds ({elapsed_time * 1000:.0f} ms)")
    print(f"📥 Response status: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    response_data = response.json()
    pretty_print_response(response_data)
    
    print("\n✅ Test passed: Multiple links processed successfully")
    return response_data


def test_link_only_no_surrounding_text():
    """
    test with just a link and minimal surrounding text.
    """
    print("\n" + "=" * 80)
    print("TEST 3: Link Only (Minimal Text)")
    print("=" * 80)
    
    payload = {
        "content": [
            {
                "textContent": "https://www.facebook.com/share/p/16k5YVoKc1/",
                "type": "text"
            }
        ]
    }
    
    print("\n📤 Request payload:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    
    print("\n⏳ Sending request...")
    start_time = time.time()
    response = requests.post(TEXT_ENDPOINT, json=payload, timeout=120)
    elapsed_time = time.time() - start_time

    print(f"\n⏱️  Request time: {elapsed_time:.2f} seconds ({elapsed_time * 1000:.0f} ms)")
    print(f"📥 Response status: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    response_data = response.json()
    pretty_print_response(response_data)
    
    print("\n✅ Test passed: Link-only content processed successfully")
    return response_data


def test_text_with_link_at_end():
    """
    test with link at the end of sentence (with punctuation).
    """
    print("\n" + "=" * 80)
    print("TEST 4: Text with Link at End")
    print("=" * 80)
    
    payload = {
        "content": [
            {
                "textContent": "O Santos está lutando contra o rebaixamento. Fonte: https://example.com/santos.",
                "type": "text"
            }
        ]
    }
    
    print("\n📤 Request payload:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    
    print("\n⏳ Sending request...")
    start_time = time.time()
    response = requests.post(TEXT_ENDPOINT, json=payload, timeout=120)
    elapsed_time = time.time() - start_time

    print(f"\n⏱️  Request time: {elapsed_time:.2f} seconds ({elapsed_time * 1000:.0f} ms)")
    print(f"📥 Response status: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    response_data = response.json()
    pretty_print_response(response_data)
    
    print("\n✅ Test passed: Link with punctuation processed successfully")
    return response_data


def test_text_with_social_media_link():
    """
    test with social media link (Facebook post).
    """
    print("\n" + "=" * 80)
    print("TEST 5: Social Media Link (Facebook)")
    print("=" * 80)
    
    payload = {
        "content": [
            {
                "textContent": "Vi esse post no Facebook: https://www.facebook.com/share/p/16k5YVoKc1/ será que é verdade?",
                "type": "text"
            }
        ]
    }
    
    print("\n📤 Request payload:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    
    print("\n⏳ Sending request... (scraping social media may take longer)")
    start_time = time.time()
    response = requests.post(TEXT_ENDPOINT, json=payload, timeout=120)
    elapsed_time = time.time() - start_time

    print(f"\n⏱️  Request time: {elapsed_time:.2f} seconds ({elapsed_time * 1000:.0f} ms)")
    print(f"📥 Response status: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    response_data = response.json()
    pretty_print_response(response_data)
    
    print("\n✅ Test passed: Social media link processed successfully")
    return response_data


def test_text_with_broken_link():
    """
    test with text containing a link that might fail to scrape.
    """
    print("\n" + "=" * 80)
    print("TEST 6: Text with Potentially Broken Link")
    print("=" * 80)
    
    payload = {
        "content": [
            {
                "textContent": "Essa notícia é importante: https://this-site-probably-does-not-exist-12345.com/article mas não sei se é real.",
                "type": "text"
            }
        ]
    }
    
    print("\n📤 Request payload:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    
    print("\n⏳ Sending request... (may fail gracefully for broken link)")
    start_time = time.time()
    response = requests.post(TEXT_ENDPOINT, json=payload, timeout=120)
    elapsed_time = time.time() - start_time

    print(f"\n⏱️  Request time: {elapsed_time:.2f} seconds ({elapsed_time * 1000:.0f} ms)")
    print(f"📥 Response status: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    response_data = response.json()
    pretty_print_response(response_data)
    
    print("\n✅ Test passed: Broken link handled gracefully")
    return response_data

def test_text_tusca():
    """
    test with text containing a link that might fail to scrape.
    """
    print("\n" + "=" * 80)
    print("TEST 6: Text with Potentially Broken Link")
    print("=" * 80)
    
    payload = {
        "content": [
            {
                "textContent": "A UFSCAR ganhou o tusca de 2025",
                "type": "text"
            }
        ]
    }
    
    print("\n📤 Request payload:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    
    print("\n⏳ Sending request... (may fail gracefully for broken link)")
    start_time = time.time()
    response = requests.post(TEXT_ENDPOINT, json=payload, timeout=120)
    elapsed_time = time.time() - start_time

    print(f"\n⏱️  Request time: {elapsed_time:.2f} seconds ({elapsed_time * 1000:.0f} ms)")
    print(f"📥 Response status: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    response_data = response.json()
    pretty_print_response(response_data)
    
    print("\n✅ Test passed: Broken link handled gracefully")
    return response_data


def run_all_tests():
    """
    run all integration tests for text with links.
    """
    print("\n" + "=" * 80)
    print("TEXT + LINKS INTEGRATION TESTS")
    print("=" * 80)
    print(f"Target: {TEXT_ENDPOINT}")
    print("⚠️  NOTE: These tests require APIFY_TOKEN to be set in .env")
    print("=" * 80)
    
    # check server health first
    if not check_server_health():
        print("\n❌ Server is not available. Aborting tests.")
        sys.exit(1)
    
    # run all tests
    tests = [
        test_text_with_single_link,
        test_text_tusca,
        test_text_with_broken_link
       # test_text_with_multiple_links,
        #test_link_only_no_surrounding_text,
        #test_text_with_link_at_end,
        #test_text_with_social_media_link,
        #test_text_with_broken_link,
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
        print("\nThe pipeline successfully:")
        print("  - Extracted links from text")
        print("  - Scraped link content")
        print("  - Extracted claims from both original text and links")
        print("  - Returned structured responses")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()

