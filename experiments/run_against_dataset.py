"""
Dataset API Testing Script

Reads dataset CSV, calls the fact-checking API concurrently,
and writes all responses to a text file.

Usage:
    python run_against_dataset.py <dataset_folder>

Example:
    python run_against_dataset.py meta_ai_2022_election
"""

import asyncio
import aiohttp
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
from dataclasses import dataclass

# ===== CONFIGURATION =====

API_BASE_URL = "http://localhost:8000"
API_ENDPOINT = "/text-without-browser"
MAX_CONCURRENT_REQUESTS = 5

# these will be set based on command-line argument
DATASET_FOLDER: Optional[Path] = None
INPUT_CSV: Optional[Path] = None
OUTPUT_FILE: Optional[Path] = None


# ===== DATA MODELS =====

@dataclass
class APIResult:
    """result of calling API for one claim"""
    claim_text: str
    success: bool
    response: str
    error_message: str = ""


# ===== API CALL FUNCTIONS =====

async def call_api(session: aiohttp.ClientSession, claim_text: str) -> APIResult:
    """
    call the fact-checking API for a single claim.

    args:
        session: aiohttp session
        claim_text: claim text to check

    returns:
        APIResult with response or error
    """
    url = f"{API_BASE_URL}{API_ENDPOINT}"

    payload = {
        "content": [
            {
                "textContent": claim_text,
                "type": "text"
            }
        ]
    }

    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
            response.raise_for_status()
            data = await response.json()
            rationale = data.get("rationale", "")

            return APIResult(
                claim_text=claim_text,
                success=True,
                response=rationale
            )
    except asyncio.TimeoutError:
        return APIResult(
            claim_text=claim_text,
            success=False,
            response="",
            error_message="Request timed out after 120 seconds"
        )
    except Exception as e:
        return APIResult(
            claim_text=claim_text,
            success=False,
            response="",
            error_message=f"API request failed: {str(e)}"
        )


async def process_claim(session: aiohttp.ClientSession, claim_text: str, index: int) -> APIResult:
    """
    process a single claim: call API and print progress.

    args:
        session: aiohttp session
        claim_text: claim to process
        index: claim number for logging

    returns:
        APIResult with outcome
    """
    print(f"[{index}] Processing: {claim_text[:60]}...")

    result = await call_api(session, claim_text)

    if result.success:
        print(f"[{index}] ✓ Success")
    else:
        print(f"[{index}] ✗ Error: {result.error_message}")

    return result


# ===== MAIN PROCESSING =====

async def process_dataset():
    """
    main function to process the entire dataset with concurrent API calls.
    """
    print("=" * 80)
    print("DATASET API TESTING")
    print("=" * 80)
    print(f"Input:  {INPUT_CSV}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"API:    {API_BASE_URL}{API_ENDPOINT}")
    print(f"Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
    print("=" * 80)

    # read CSV and extract claim texts
    print("\nReading CSV...")

    import csv
    claims = []

    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        # try common column names for claim text
        first_row = next(reader)
        f.seek(0)
        next(reader)  # skip header again

        # detect which column has the claim text
        claim_column = None
        for col in ["Título da checagem", "claim_text", "text", "claim", "Claim"]:
            if col in first_row:
                claim_column = col
                break

        if not claim_column:
            print(f"❌ Error: Could not find claim text column in CSV")
            print(f"   Available columns: {', '.join(first_row.keys())}")
            sys.exit(1)

        print(f"✓ Using column '{claim_column}' for claim text")

        # reset and read all claims
        f.seek(0)
        reader = csv.DictReader(f)

        for row in reader:
            claim_text = row.get(claim_column, "").strip()
            if claim_text:
                claims.append(claim_text)

    print(f"Loaded {len(claims)} claims from CSV")

    # process claims with concurrency limit
    print(f"\nProcessing with max {MAX_CONCURRENT_REQUESTS} concurrent requests...\n")
    results = []

    # create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def process_with_semaphore(session: aiohttp.ClientSession, claim_text: str, index: int):
        async with semaphore:
            return await process_claim(session, claim_text, index)

    # create session and process all claims
    async with aiohttp.ClientSession() as session:
        tasks = [process_with_semaphore(session, claim, i+1) for i, claim in enumerate(claims)]
        results = await asyncio.gather(*tasks)

    # write results to text file
    print(f"\nWriting results to {OUTPUT_FILE}...")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("DATASET API TESTING RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Dataset: {INPUT_CSV.name}\n")
        f.write(f"API: {API_BASE_URL}{API_ENDPOINT}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total claims: {len(results)}\n")
        f.write("=" * 80 + "\n\n")

        for idx, result in enumerate(results, 1):
            f.write(f"{'=' * 80}\n")
            f.write(f"CLAIM {idx}\n")
            f.write(f"{'=' * 80}\n\n")
            f.write(f"Text: {result.claim_text}\n\n")

            if result.success:
                f.write(f"Status: SUCCESS\n\n")
                f.write(f"Response:\n")
                f.write(f"{'-' * 80}\n")
                f.write(result.response)
                f.write(f"\n{'-' * 80}\n\n")
            else:
                f.write(f"Status: ERROR\n\n")
                f.write(f"Error: {result.error_message}\n\n")

    # summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    successful = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)
    total = len(results)

    print(f"Total processed:  {total}")
    print(f"✓ Successful:     {successful} ({successful/total*100:.1f}%)")
    print(f"✗ Failed:         {failed} ({failed/total*100:.1f}%)")
    print("=" * 80)
    print(f"\nResults saved to: {OUTPUT_FILE}")


# ===== CONFIGURATION SETUP =====

def setup_paths(dataset_folder_name: str):
    """
    setup file paths based on dataset folder name.

    args:
        dataset_folder_name: name of the dataset folder (e.g., "meta_ai_2022_election")
    """
    global DATASET_FOLDER, INPUT_CSV, OUTPUT_FILE

    script_dir = Path(__file__).parent
    DATASET_FOLDER = script_dir / dataset_folder_name

    # check if folder exists
    if not DATASET_FOLDER.exists():
        print(f"❌ Error: Dataset folder not found: {DATASET_FOLDER}")
        print(f"   Looking in: {script_dir}")
        print(f"   Available folders:")
        for item in script_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.') and item.name != 'venv':
                print(f"     - {item.name}")
        sys.exit(1)

    # look for CSV files in the folder
    csv_files = sorted(DATASET_FOLDER.glob("*.csv"))

    if not csv_files:
        print(f"❌ Error: No CSV files found in {DATASET_FOLDER}")
        sys.exit(1)

    # if multiple CSVs, list them and use the first one alphabetically
    if len(csv_files) > 1:
        print(f"⚠️  Multiple CSV files found:")
        for idx, csv_file in enumerate(csv_files, 1):
            print(f"     {idx}. {csv_file.name}")
        INPUT_CSV = csv_files[0]
        print(f"     Using: {INPUT_CSV.name}")
    else:
        INPUT_CSV = csv_files[0]

    # setup output path in the same folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    OUTPUT_FILE = DATASET_FOLDER / f"api_responses_{timestamp}.txt"

    print(f"✓ Dataset folder: {DATASET_FOLDER.name}")
    print(f"✓ Input CSV:      {INPUT_CSV.name}")
    print(f"✓ Output file:    {OUTPUT_FILE.name}")


# ===== ENTRY POINT =====

def main():
    """entry point for the script"""

    # parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python run_against_dataset.py <dataset_folder>")
        print("\nExample:")
        print("  python run_against_dataset.py meta_ai_2022_election")
        print("\nAvailable dataset folders:")
        script_dir = Path(__file__).parent
        for item in script_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.') and item.name != 'venv':
                print(f"  - {item.name}")
        sys.exit(1)

    dataset_folder = sys.argv[1]
    setup_paths(dataset_folder)

    asyncio.run(process_dataset())


if __name__ == "__main__":
    main()
