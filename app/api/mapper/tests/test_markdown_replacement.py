"""
quick test to verify markdown link replacement works correctly.

tests that markdown-style links [text](url) are replaced with just url,
removing the markdown formatting entirely.

run with: python -m app.api.mapper.tests.test_markdown_replacement
"""
from ..formating import replace_markdown_links

def test_replace_markdown_links():
    """
    test the markdown link replacement function.

    verifies that [text](url) is replaced with url.
    """

    test_cases = [
        # (input, expected_output)
        (
            "Check [fifa.com](https://www.fifa.com/pt/articles/ronaldo-pele) for more info",
            "Check https://www.fifa.com/pt/articles/ronaldo-pele for more info"
        ),
        (          "https://www.fifa.com/pt/articles/ronaldo-pele for more info",
            "https://www.fifa.com/pt/articles/ronaldo-pele for more info"
        ),
         (          "(https://www.fifa.com/pt/articles/ronaldo-pele) for more info",
            "(https://www.fifa.com/pt/articles/ronaldo-pele) for more info"
        ),
         (          "[https://www.fifa.com/pt/articles/ronaldo-pele] for more info",
            "[https://www.fifa.com/pt/articles/ronaldo-pele] for more info"
        ),
        (
            "Multiple sources: [source1](url1) and [source2](url2) here",
            "Multiple sources: url1 and url2 here"
        ),
        (
            "No links here, just plain text",
            "No links here, just plain text"
        ),
        (
            "Mixed: [link](url) and plain http://example.com urls",
            "Mixed: url and plain http://example.com urls"
        ),
        (
            "Mixed: [link] urls",
            "Mixed: [link] urls"
        ),
        (
            "Mixed: (link) urls",
            "Mixed: (link) urls"
        ),
        (
            "Brackets and parens separated: [link] (url) should not match",
            "Brackets and parens separated: [link] (url) should not match"
        ),
        (
            "Reversed order: (url)[text] should not match",
            "Reversed order: (url)[text] should not match"
        ),
        (
            "Multiple brackets: [link1] [link2] should stay",
            "Multiple brackets: [link1] [link2] should stay"
        ),
        (
            "Multiple parens: (url1) (url2) should stay",
            "Multiple parens: (url1) (url2) should stay"
        ),
        (
            "Empty brackets: [](url) edge case",
            "Empty brackets: url edge case"
        ),
        (
            "Empty parens: [text]() edge case",
            "Empty parens:  edge case"
        ),
        (
            "Link text with spaces: [link with spaces](http://example.com)",
            "Link text with spaces: http://example.com"
        ),
        (
            "URL with query params: [click](http://site.com?param=val&other=123)",
            "URL with query params: http://site.com?param=val&other=123"
        ),
        (
            "URL with fragment: [section](http://example.com#section-1)",
            "URL with fragment: http://example.com#section-1"
        ),
        (
            "[Start](url) of string",
            "url of string"
        ),
        (
            "End of string [link](url)",
            "End of string url"
        ),
        (
            "Consecutive [first](url1)[second](url2) links",
            "Consecutive url1url2 links"
        ),
        (
            "Nested [[brackets]](url) pattern",
            "Nested [brackets] pattern"
        ),
        (
            "Nested parens [text]((nested))",
            "Nested parens (nested)"
        ),
        (
            "Only opening bracket [ without closing",
            "Only opening bracket [ without closing"
        ),
        (
            "Only opening paren ( without closing",
            "Only opening paren ( without closing"
        ),
        (
            "([fifa.com](https://www.fifa.com/pt/articles/ronaldo-pele-ademir-jairzinho-vava-os-5-maiores-artilheiros-do-brasil-na-historia-das-copas-do-mundo?utm_source=openai))",
            "(https://www.fifa.com/pt/articles/ronaldo-pele-ademir-jairzinho-vava-os-5-maiores-artilheiros-do-brasil-na-historia-das-copas-do-mundo?utm_source=openai)"
        ),
    ]

    print("Testing markdown link replacement:\n")
    print("=" * 80)

    passed = 0
    failed = 0

    for i, (input_text, expected) in enumerate(test_cases, 1):
        result = replace_markdown_links(input_text)
        status = "✓" if result == expected else "✗"

        if result == expected:
            passed += 1
        else:
            failed += 1

        print(f"\nTest {i}: {status}")
        print(f"Input:    {input_text}")
        print(f"Expected: {expected}")
        print(f"Got:      {result}")

        if result != expected:
            print("  ⚠️  MISMATCH!")

    print("\n" + "=" * 80)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 80)

    return failed == 0

if __name__ == "__main__":
    success = test_replace_markdown_links()
    exit(0 if success else 1)
