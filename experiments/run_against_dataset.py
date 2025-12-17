"""
Dataset Validation Script

Reads dataset CSV, calls the fact-checking API concurrently,
and validates that verdicts match expected values.

Usage:
    python run_against_dataset.py <dataset_folder>

Example:
    python run_against_dataset.py meta_ai_2022_election
"""

import csv
import json
import asyncio
import aiohttp
import sys
from pathlib import Path
from typing import List, Dict, Literal, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

# ===== CONFIGURATION =====

API_BASE_URL = "http://localhost:8000"
API_ENDPOINT = "/text-adjudication-search"
MAX_CONCURRENT_REQUESTS = 5

# these will be set based on command-line argument
DATASET_FOLDER: Optional[Path] = None
INPUT_CSV: Optional[Path] = None
OUTPUT_CSV: Optional[Path] = None
DATASET_CONFIG: Optional[Dict] = None

VerdictType = Literal["Verdadeiro", "Falso", "Fora de Contexto", "Fontes insuficientes para verificar"]

# Default verdict mapping (can be overridden by dataset config)
DEFAULT_VERDICT_MAPPING = {
    "Falsa": "Falso",
    "Verdadeira": "Verdadeiro",
    "Fora de contexto": "Fora de Contexto",
    "Insuficiente": "Fontes insuficientes para verificar",
}

# Default column mapping (can be overridden by dataset config)
DEFAULT_COLUMN_MAPPING = {
    "claim_text": "Título da checagem",
    "expected_verdict": "Natureza da notícia",
    "date": "Data da checagem",
    "metadata1": "Candidato(s) favorecidos(s) pela notícia falsa",
    "metadata2": "Agência",
    "link": "Link"
}


# ===== DATA MODELS =====

@dataclass
class DatasetRow:
    """represents a row from the dataset CSV"""
    claim_text: str
    expected_verdict: str
    date: str = ""
    metadata1: str = ""
    metadata2: str = ""
    link: str = ""


@dataclass
class ValidationResult:
    """result of validating one claim"""
    titulo: str
    expected_verdicts: List[str]  # changed to list to support multiple acceptable verdicts
    raw_response: str
    validation_status: Literal["passed", "failed", "error"]
    error_message: str = ""
    found_verdicts: List[str] = field(default_factory=list)


# ===== HELPER FUNCTIONS =====

def load_dataset_config(dataset_folder: Path) -> Dict:
    """
    load dataset configuration from dataset_config.json if it exists.

    the config file can specify:
    - column_mapping: dict mapping standard fields to CSV column names
    - verdict_mapping: dict mapping CSV verdict values to API format

    args:
        dataset_folder: path to dataset folder

    returns:
        configuration dict (empty if no config file exists)
    """
    config_file = dataset_folder / "dataset_config.json"

    if not config_file.exists():
        print(f"⚠️  No dataset_config.json found, using default E-farsas column mapping")
        return {}

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✓ Loaded dataset config from {config_file.name}")
        return config
    except Exception as e:
        print(f"⚠️  Error loading dataset config: {e}, using defaults")
        return {}


def normalize_verdict(csv_verdict: str) -> List[str]:
    """
    normalize verdict from CSV to API format.

    supports both single verdict and multiple acceptable verdicts.

    args:
        csv_verdict: verdict string from CSV (e.g., "Falsa", "fake")

    returns:
        list of acceptable verdicts (e.g., ["Falso"] or ["Falso", "Fora de Contexto"])
    """
    verdict_mapping = DATASET_CONFIG.get("verdict_mapping", DEFAULT_VERDICT_MAPPING) if DATASET_CONFIG else DEFAULT_VERDICT_MAPPING
    mapped = verdict_mapping.get(csv_verdict, csv_verdict)

    # support both string and list of strings in config
    if isinstance(mapped, list):
        return mapped
    else:
        return [mapped]


def extract_verdicts_from_response(response_text: str) -> List[str]:
    """
    extract all verdict strings from the API response.

    looks for occurrences of the four possible verdicts in the response text.

    args:
        response_text: raw API response text

    returns:
        list of found verdicts
    """
    possible_verdicts: List[VerdictType] = [
        "Verdadeiro",
        "Falso",
        "Fora de Contexto",
        "Fontes insuficientes para verificar"
    ]

    found = []
    for verdict in possible_verdicts:
        if verdict in response_text:
            found.append(verdict)

    return found #type: ignore


def validate_response(expected_verdicts: List[str], response_text: str) -> Tuple[bool, List[str]]:
    """
    validate that response contains only acceptable verdicts.

    args:
        expected_verdicts: list of acceptable verdicts (e.g., ["Falso"] or ["Falso", "Fora de Contexto"])
        response_text: raw API response

    returns:
        tuple of (is_valid, list_of_found_verdicts)
    """
    found_verdicts = extract_verdicts_from_response(response_text)

    # check that:
    # 1. at least one instance of an acceptable verdict exists
    # 2. no other verdicts (outside acceptable list) are present
    has_acceptable = any(v in expected_verdicts for v in found_verdicts)
    only_acceptable = all(v in expected_verdicts for v in found_verdicts)

    is_valid = has_acceptable and only_acceptable
    return is_valid, found_verdicts


# ===== API CALL FUNCTIONS =====

async def call_api(session: aiohttp.ClientSession, titulo: str) -> Dict:
    """
    call the fact-checking API for a single claim.

    args:
        session: aiohttp session
        titulo: claim text to check

    returns:
        API response as dict
    """
    url = f"{API_BASE_URL}{API_ENDPOINT}"

    payload = {
        "content": [
            {
                "textContent": titulo,
                "type": "text"
            }
        ]
    }

    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
            response.raise_for_status()
            return await response.json()
    except asyncio.TimeoutError:
        raise Exception("Request timed out after 120 seconds")
    except aiohttp.ClientError as e:
        raise Exception(f"API request failed: {str(e)}")


async def process_row(session: aiohttp.ClientSession, row: DatasetRow) -> ValidationResult:
    """
    process a single dataset row: call API and validate response.

    args:
        session: aiohttp session
        row: dataset row to process

    returns:
        ValidationResult with outcome
    """
    print(f"Processing: {row.claim_text[:60]}...")

    expected_verdicts = normalize_verdict(row.expected_verdict)

    try:
        # call API
        response_data = await call_api(session, row.claim_text)
        raw_response = response_data.get("rationale", "")

        # validate response
        is_valid, found_verdicts = validate_response(expected_verdicts, raw_response)

        if is_valid:
            verdicts_str = " or ".join(expected_verdicts) if len(expected_verdicts) > 1 else expected_verdicts[0]
            print(f"   PASSED - Found acceptable verdict: {verdicts_str}")
            return ValidationResult(
                titulo=row.claim_text,
                expected_verdicts=expected_verdicts,
                raw_response=raw_response,
                validation_status="passed",
                found_verdicts=found_verdicts
            )
        else:
            verdicts_str = " or ".join(expected_verdicts) if len(expected_verdicts) > 1 else expected_verdicts[0]
            print(f"   FAILED - Expected: {verdicts_str}, Found: {found_verdicts}")
            return ValidationResult(
                titulo=row.claim_text,
                expected_verdicts=expected_verdicts,
                raw_response=raw_response,
                validation_status="failed",
                error_message=f"Expected one of {expected_verdicts}, but found: {found_verdicts}",
                found_verdicts=found_verdicts
            )

    except Exception as e:
        print(f"   ERROR - {str(e)}")
        return ValidationResult(
            titulo=row.claim_text,
            expected_verdicts=expected_verdicts,
            raw_response="",
            validation_status="error",
            error_message=str(e)
        )


# ===== MAIN PROCESSING =====

async def process_dataset():
    """
    main function to process the entire dataset with concurrent API calls.
    """
    print("=" * 80)
    print("E-FARSAS DATASET VALIDATION")
    print("=" * 80)
    print(f"Input:  {INPUT_CSV}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"API:    {API_BASE_URL}{API_ENDPOINT}")
    print(f"Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
    print("=" * 80)

    # read CSV
    print("\nReading CSV...")
    rows: List[DatasetRow] = []

    # get column mapping from config or use defaults
    column_mapping = DATASET_CONFIG.get("column_mapping", DEFAULT_COLUMN_MAPPING) if DATASET_CONFIG else DEFAULT_COLUMN_MAPPING

    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row_dict in reader:
            rows.append(DatasetRow(
                claim_text=row_dict.get(column_mapping["claim_text"], ""),
                expected_verdict=row_dict.get(column_mapping["expected_verdict"], ""),
                date=row_dict.get(column_mapping.get("date", ""), ""),
                metadata1=row_dict.get(column_mapping.get("metadata1", ""), ""),
                metadata2=row_dict.get(column_mapping.get("metadata2", ""), ""),
                link=row_dict.get(column_mapping.get("link", ""), "")
            ))

    print(f"Loaded {len(rows)} rows from CSV")

    # process rows with concurrency limit
    print(f"\nProcessing with max {MAX_CONCURRENT_REQUESTS} concurrent requests...")
    results: List[ValidationResult] = []

    # create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def process_with_semaphore(session: aiohttp.ClientSession, row: DatasetRow):
        async with semaphore:
            return await process_row(session, row)

    # create session and process all rows
    async with aiohttp.ClientSession() as session:
        tasks = [process_with_semaphore(session, row) for row in rows]
        results = await asyncio.gather(*tasks)

    # write results to CSV
    print(f"\nWriting results to {OUTPUT_CSV}...")

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # header
        writer.writerow([
            "T�tulo da checagem",
            "Expected Verdict(s)",
            "Validation Status",
            "Found Verdicts",
            "Error Message"
        ])

        # data rows
        for result in results:
            writer.writerow([
                result.titulo,
                " or ".join(result.expected_verdicts),  # join multiple acceptable verdicts
                result.validation_status,
                ", ".join(result.found_verdicts) if result.found_verdicts else "",
                result.error_message
            ])

    # write raw responses to separate text file
    raw_responses_file = OUTPUT_CSV.parent / f"raw_responses_{OUTPUT_CSV.stem.replace('results_', '')}.txt"
    print(f"Writing raw responses to {raw_responses_file}...")

    with open(raw_responses_file, 'w', encoding='utf-8') as f:
        for idx, result in enumerate(results, 1):
            f.write(f"========================================\n")
            f.write(f"Response {idx}\n")
            f.write(f"Claim: {result.titulo[:100]}...\n")
            f.write(f"Expected: {' or '.join(result.expected_verdicts)}\n")
            f.write(f"Status: {result.validation_status}\n")
            f.write(f"========================================\n\n")
            f.write(result.raw_response if result.raw_response else "(No response)")
            f.write("\n\n")
            f.write("--------\n\n")

    # summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in results if r.validation_status == "passed")
    failed = sum(1 for r in results if r.validation_status == "failed")
    errors = sum(1 for r in results if r.validation_status == "error")
    total = len(results)

    print(f"Total processed:  {total}")
    print(f" Passed:         {passed} ({passed/total*100:.1f}%)")
    print(f" Failed:         {failed} ({failed/total*100:.1f}%)")
    print(f"� Errors:         {errors} ({errors/total*100:.1f}%)")
    print("=" * 80)
    print(f"\nResults saved to:")
    print(f"  CSV:            {OUTPUT_CSV}")
    print(f"  Raw responses:  {raw_responses_file}")


# ===== CONFIGURATION SETUP =====

def setup_paths(dataset_folder_name: str):
    """
    setup file paths based on dataset folder name.

    looks for CSV file in the dataset folder and sets up output path.
    also loads dataset configuration if available.

    args:
        dataset_folder_name: name of the dataset folder (e.g., "meta_ai_2022_election")
    """
    global DATASET_FOLDER, INPUT_CSV, OUTPUT_CSV, DATASET_CONFIG

    script_dir = Path(__file__).parent
    DATASET_FOLDER = script_dir / dataset_folder_name

    # check if folder exists
    if not DATASET_FOLDER.exists():
        print(f"❌ Error: Dataset folder not found: {DATASET_FOLDER}")
        print(f"   Looking in: {script_dir}")
        print(f"   Available folders:")
        for item in script_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                print(f"     - {item.name}")
        sys.exit(1)

    # look for CSV files in the folder
    csv_files = sorted(DATASET_FOLDER.glob("*.csv"))  # sort alphabetically for consistency

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
    OUTPUT_CSV = DATASET_FOLDER / f"results_{timestamp}.csv"

    # load dataset configuration
    DATASET_CONFIG = load_dataset_config(DATASET_FOLDER)

    print(f"✓ Dataset folder: {DATASET_FOLDER.name}")
    print(f"✓ Input CSV:      {INPUT_CSV.name}")
    print(f"✓ Output CSV:     {OUTPUT_CSV.name}")


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
