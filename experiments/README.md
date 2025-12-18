# Dataset API Testing

This folder contains a simple script to test the fact-checking API against datasets.

## Structure

```
experiments/
├── run_against_dataset.py        # Main testing script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── meta_ai_2025_fake_news_g1/    # Another dataset
    └── ...
```

## Setup

1. Create and activate virtual environment:
```bash
cd experiments
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure the API server is running:
```bash
# In another terminal, from project root:
uvicorn app.main:app --reload --port 8000
```

## Usage

Run the script with a dataset folder:

```bash
# Run against another dataset
python run_against_dataset.py meta_ai_2025_fake_news_g1
```

**No arguments?** The script will show available dataset folders:
```bash
python run_against_dataset.py
```

## Configuration

Edit constants in `run_against_dataset.py`:

```python
API_BASE_URL = "http://localhost:8000"
API_ENDPOINT = "/text"
MAX_CONCURRENT_REQUESTS = 5
```

## How It Works

1. **Reads CSV** - Finds the first `.csv` file in the dataset folder
2. **Auto-detects claim column** - Looks for common column names like "Título da checagem", "claim_text", "text", "claim"
3. **Calls API concurrently** - Max 5 requests in parallel
4. **Writes responses** - Saves all API responses to a text file

## Dataset Selection

When you run the script, it:
1. Looks in the specified dataset folder for `.csv` files
2. If multiple CSVs exist, lists them and uses the **first one alphabetically**
3. Auto-detects which column contains the claim text
4. Writes results to `api_responses_YYYYMMDD_HHMMSS.txt` in the **same dataset folder**

**Tip:** If you have multiple CSV files and want a specific one to be used, either:
- Name it so it comes first alphabetically (e.g., `01_dataset.csv`)
- Move or delete the other CSV files temporarily

## Output File Format

The script generates a text file with all API responses:

```
================================================================================
DATASET API TESTING RESULTS
================================================================================
Dataset: filtered_dataset.csv
API: http://localhost:8000/text
Date: 2025-12-17 15:30:45
Total claims: 145
================================================================================

================================================================================
CLAIM 1
================================================================================

Text: Sergio Moro tirou foto ao lado do Lula após seu partido...

Status: SUCCESS

Response:
--------------------------------------------------------------------------------
[Full API response with verdict, rationale, and citations]
--------------------------------------------------------------------------------

================================================================================
CLAIM 2
================================================================================

...
```

## Example Console Output

```
================================================================================
DATASET API TESTING
================================================================================
Input:  /path/to/filtered_dataset.csv
Output: /path/to/api_responses_20251217_153045.txt
API:    http://localhost:8000/text
Max concurrent requests: 5
================================================================================

Reading CSV...
✓ Using column 'Título da checagem' for claim text
Loaded 145 claims from CSV

Processing with max 5 concurrent requests...

[1] Processing: Sergio Moro tirou foto ao lado do Lula após seu partido...
[2] Processing: A vacina X causa infertilidade...
[1] ✓ Success
[3] Processing: O governo vai aumentar impostos...
[2] ✓ Success
...

================================================================================
SUMMARY
================================================================================
Total processed:  145
✓ Successful:     142 (97.9%)
✗ Failed:         3 (2.1%)
================================================================================

Results saved to: api_responses_20251217_153045.txt
```
