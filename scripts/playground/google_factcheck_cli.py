#!/usr/bin/env python3
"""
interactive CLI tool for testing Google Fact-Check API.

allows users to interactively query the Google Fact-Check API,
view detailed responses, and test various claims.

usage:
    python google_factcheck_cli.py
"""

import os
import sys
import asyncio
from pathlib import Path

# add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.playground.common import (
    Menu,
    print_header,
    print_section,
    print_success,
    print_error,
    print_warning,
    print_info,
    print_json,
    print_dict_table,
    print_citation_list,
    prompt_input,
    prompt_yes_no,
    prompt_choice,
    prompt_multiline,
    with_spinner,
    handle_api_error,
    Colors,
)

from app.ai.context.factcheckapi import GoogleFactCheckGatherer
from app.models import ExtractedClaim, ClaimSource


# ===== PREDEFINED TEST CLAIMS =====

EXAMPLE_CLAIMS = {
    "Vaccines cause autism (English)": "Vaccines cause autism",
    "Climate change is a hoax (English)": "Climate change is a hoax",
    "Earth is flat (English)": "The Earth is flat",
    "COVID-19 vaccines contain microchips (English)": "COVID-19 vaccines contain microchips",
    "5G causes cancer (English)": "5G technology causes cancer",
    "Eleição foi fraudada (Portuguese)": "A eleição presidencial de 2022 foi fraudada",
    "Vacinas causam autismo (Portuguese)": "Vacinas causam autismo em crianças",
    "Terra é plana (Portuguese)": "A Terra é plana e não esférica",
}


# ===== GLOBAL STATE =====

gatherer: GoogleFactCheckGatherer = None
last_result = None


# ===== INITIALIZATION =====

def initialize_gatherer() -> bool:
    """initialize the Google Fact-Check API gatherer."""
    global gatherer

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print_error("GOOGLE_API_KEY environment variable not set")
        print_info("Please set it with: export GOOGLE_API_KEY='your-key-here'")
        return False

    try:
        gatherer = GoogleFactCheckGatherer(api_key=api_key, max_results=10)
        print_success("Google Fact-Check API gatherer initialized")
        return True
    except Exception as e:
        print_error(f"Failed to initialize gatherer: {e}")
        return False


# ===== API OPERATIONS =====

async def query_claim_async(claim_text: str, verbose: bool = True) -> list:
    """query the Google Fact-Check API for a claim."""
    global last_result

    claim = ExtractedClaim(
        id="cli-test-claim",
        text=claim_text,
        source=ClaimSource(source_type="original_text", source_id="cli-test"),
        entities=[],
    )

    if verbose:
        print_section("Querying Google Fact-Check API")
        print(f"Claim: {Colors.BOLD}{claim_text}{Colors.END}")
        print()

    try:
        citations = await gatherer.gather(claim)
        last_result = citations
        return citations
    except Exception as e:
        handle_api_error(e, "Google Fact-Check API")
        return []


def query_claim(claim_text: str, verbose: bool = True) -> list:
    """synchronous wrapper for querying claims."""
    return asyncio.run(query_claim_async(claim_text, verbose))


# ===== MENU HANDLERS =====

def handle_query_custom_claim():
    """handle custom claim query."""
    print_header("Query Custom Claim")

    claim_text = prompt_input("Enter claim to fact-check")

    if not claim_text:
        print_warning("No claim entered")
        return

    citations = query_claim(claim_text)

    print_section("Results")
    print_citation_list(citations)

    if citations:
        if prompt_yes_no("\nShow full citation details?", default=False):
            for i, citation in enumerate(citations, 1):
                print(f"\n{Colors.BOLD}Citation {i} - Full Details:{Colors.END}")
                citation_dict = {
                    "title": citation.title,
                    "publisher": citation.publisher,
                    "url": citation.url,
                    "rating": citation.rating,
                    "rating_comment": citation.rating_comment,
                    "date": citation.date,
                    "source": citation.source,
                    "citation_text": citation.citation_text,
                }
                print_dict_table(citation_dict)


def handle_query_example_claim():
    """handle example claim query."""
    print_header("Query Example Claim")

    claim_label = prompt_choice(
        "Select an example claim:",
        list(EXAMPLE_CLAIMS.keys())
    )

    claim_text = EXAMPLE_CLAIMS[claim_label]

    print_info(f"Selected: {claim_text}")
    print()

    citations = query_claim(claim_text)

    print_section("Results")
    print_citation_list(citations)


def handle_batch_query():
    """handle batch query of multiple claims."""
    print_header("Batch Query Multiple Claims")

    print_info("Enter claims one per line (type 'DONE' on a new line to finish)")
    print()

    claims = []
    while True:
        claim = input(f"{len(claims) + 1}. ").strip()
        if claim.upper() == "DONE":
            break
        if claim:
            claims.append(claim)

    if not claims:
        print_warning("No claims entered")
        return

    print_section(f"Processing {len(claims)} claims")

    results = []
    for i, claim_text in enumerate(claims, 1):
        print(f"\n{Colors.BOLD}[{i}/{len(claims)}] {claim_text}{Colors.END}")

        citations = query_claim(claim_text, verbose=False)
        results.append((claim_text, citations))

        if citations:
            print_success(f"Found {len(citations)} citation(s)")
        else:
            print_warning("No citations found")

    # summary
    print_section("Batch Query Summary")

    total_citations = sum(len(cits) for _, cits in results)
    claims_with_results = sum(1 for _, cits in results if cits)

    print(f"Total claims: {len(claims)}")
    print(f"Claims with results: {claims_with_results}")
    print(f"Total citations: {total_citations}")

    if prompt_yes_no("\nShow detailed results?", default=True):
        for claim_text, citations in results:
            print(f"\n{Colors.BOLD}Claim: {claim_text}{Colors.END}")
            print_citation_list(citations)


def handle_multiline_claim():
    """handle multiline claim query."""
    print_header("Query Multiline Claim")

    claim_text = prompt_multiline(
        "Enter your claim (can be multiple lines):",
        end_marker="END"
    )

    if not claim_text.strip():
        print_warning("No claim entered")
        return

    print_info(f"Claim entered ({len(claim_text)} characters)")

    citations = query_claim(claim_text)

    print_section("Results")
    print_citation_list(citations)


def handle_view_last_result():
    """view the last query result in detail."""
    if last_result is None:
        print_warning("No previous query results available")
        return

    print_header("Last Query Results")
    print_citation_list(last_result)

    if last_result and prompt_yes_no("\nShow as JSON?", default=False):
        print()
        citations_dict = [
            {
                "title": c.title,
                "publisher": c.publisher,
                "url": c.url,
                "rating": c.rating,
                "rating_comment": c.rating_comment,
                "date": c.date,
                "source": c.source,
            }
            for c in last_result
        ]
        print_json(citations_dict)


def handle_test_api_connection():
    """test the API connection with a simple query."""
    print_header("Test API Connection")

    test_claim = "COVID-19 vaccines are safe"
    print_info(f"Testing with claim: '{test_claim}'")
    print()

    try:
        citations = query_claim(test_claim)

        if citations:
            print_success("API connection successful!")
            print_success(f"Returned {len(citations)} citation(s)")
        else:
            print_warning("API connection successful, but no citations found")
            print_info("This is normal for claims without indexed fact-checks")

    except Exception as e:
        print_error("API connection failed")
        handle_api_error(e, "Google Fact-Check API")


def handle_show_config():
    """show current configuration."""
    print_header("Current Configuration")

    config = {
        "API Key": "***" + (gatherer.api_key[-4:] if gatherer and gatherer.api_key else "not set"),
        "Max Results": gatherer.max_results if gatherer else "N/A",
        "Timeout": f"{gatherer.timeout}s" if gatherer else "N/A",
        "Base URL": gatherer.base_url if gatherer else "N/A",
    }

    print_dict_table(config)


def handle_compare_languages():
    """compare results for same claim in different languages."""
    print_header("Compare Language Results")

    claim_en = "Vaccines cause autism"
    claim_pt = "Vacinas causam autismo"

    print_section("Querying English claim")
    citations_en = query_claim(claim_en, verbose=False)
    print_success(f"English: {len(citations_en)} citation(s)")

    print_section("Querying Portuguese claim")
    citations_pt = query_claim(claim_pt, verbose=False)
    print_success(f"Portuguese: {len(citations_pt)} citation(s)")

    print_section("Comparison Summary")
    print(f"English results: {len(citations_en)}")
    print(f"Portuguese results: {len(citations_pt)}")

    if citations_en:
        print(f"\n{Colors.BOLD}English citations:{Colors.END}")
        print_citation_list(citations_en)

    if citations_pt:
        print(f"\n{Colors.BOLD}Portuguese citations:{Colors.END}")
        print_citation_list(citations_pt)


# ===== MAIN =====

def main():
    """main entry point."""
    print_header("Google Fact-Check API Interactive CLI")

    # initialize
    if not initialize_gatherer():
        sys.exit(1)

    # create menu
    menu = Menu("Google Fact-Check API Tester")

    menu.add_option("Query custom claim", handle_query_custom_claim)
    menu.add_option("Query example claim", handle_query_example_claim)
    menu.add_option("Query multiline claim", handle_multiline_claim)
    menu.add_option("Batch query multiple claims", handle_batch_query)

    menu.add_separator()

    menu.add_option("View last query results", handle_view_last_result)
    menu.add_option("Compare language results", handle_compare_languages)

    menu.add_separator()

    menu.add_option("Test API connection", handle_test_api_connection)
    menu.add_option("Show configuration", handle_show_config)

    # run interactive loop
    try:
        menu.run()
    except KeyboardInterrupt:
        print_info("\n\nExiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()
