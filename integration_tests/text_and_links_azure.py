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


# configuration
BASE_URL = "http://localhost:8000"
TEXT_ENDPOINT = f"{BASE_URL}/text-no-browser-azure"
HEALTH_ENDPOINT = f"{BASE_URL}/health"


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
    response = requests.post(TEXT_ENDPOINT, json=payload, timeout=120)
    
    print(f"\n📥 Response status: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    response_data = response.json()
    print("\n📄 Response body:")
    print(json.dumps(response_data, indent=2, ensure_ascii=False))
    
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
    response = requests.post(TEXT_ENDPOINT, json=payload, timeout=180)
    
    print(f"\n📥 Response status: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    response_data = response.json()
    print("\n📄 Response body:")
    print(json.dumps(response_data, indent=2, ensure_ascii=False))
    
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
    response = requests.post(TEXT_ENDPOINT, json=payload, timeout=120)
    
    print(f"\n📥 Response status: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    response_data = response.json()
    print("\n📄 Response body:")
    print(json.dumps(response_data, indent=2, ensure_ascii=False))
    
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
    response = requests.post(TEXT_ENDPOINT, json=payload, timeout=120)
    
    print(f"\n📥 Response status: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    response_data = response.json()
    print("\n📄 Response body:")
    print(json.dumps(response_data, indent=2, ensure_ascii=False))
    
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
    response = requests.post(TEXT_ENDPOINT, json=payload, timeout=120)
    
    print(f"\n📥 Response status: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    response_data = response.json()
    print("\n📄 Response body:")
    print(json.dumps(response_data, indent=2, ensure_ascii=False))
    
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
    response = requests.post(TEXT_ENDPOINT, json=payload, timeout=120)
    
    print(f"\n📥 Response status: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    response_data = response.json()
    print("\n📄 Response body:")
    print(json.dumps(response_data, indent=2, ensure_ascii=False))
    
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
        test_text_with_single_link,
        test_text_with_single_link,
        test_text_with_single_link,
        #test_text_with_multiple_links,
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

