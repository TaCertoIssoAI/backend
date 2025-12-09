# Image Claim Extraction Test Script

This script compares two approaches for extracting fact-checkable claims from images using GPT-4o.

## Overview

The script tests two different approaches:

### Approach 1: Direct Image Claim Extraction
- Sends the image directly to GPT-4o with the claim extraction prompt
- Uses the same system prompt as the production pipeline (`IMAGE_CLAIM_EXTRACTION_SYSTEM_PROMPT`)
- Returns structured claims with entities and justifications

### Approach 2: Transcribe First, Then Extract
- **Step 1**: Sends the image to GPT-4o with a transcription prompt that focuses on:
  - Transcribing all text with visual formatting details (bold, caps, colors, etc.)
  - Describing visual elements, especially famous people, politicians, or celebrities
  - Noting unusual details that might help with fact-checking
- **Step 2**: Passes the transcription to the claim extractor (text-only, no image)
  - Uses the existing `extract_claims()` function from the pipeline
  - Configured with GPT-4o model

## Setup

1. **Install dependencies** (if not already installed):
   ```bash
   pip install langchain-openai langchain-core pydantic
   ```

2. **Set OpenAI API key**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. **Add test images**:
   - Place your test images in the `scripts/images/` folder
   - Supported formats: `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.webp`

## Usage

Run the script from the project root:

```bash
python3 scripts/image_claim_extraction.py
```

## Output

For each image, the script will print:

### Approach 1 Output:
```
APPROACH 1: Direct Image Claim Extraction
Image: example.png
================================================================================

Extracted 2 claims:

Claim 1:
  Text: [Normalized claim text]
  Entities: [list of entities]
  LLM Comment: [Why this is fact-checkable]

Claim 2:
  ...
```

### Approach 2 Output:
```
APPROACH 2: Transcribe Image Then Extract Claims
Image: example.png
================================================================================

Step 1: Transcribing image...
Transcription:
[Full transcription with visual details and context]

Step 2: Extracting claims from transcription...
Extracted 2 claims:

Claim 1:
  ID: [UUID]
  Text: [Normalized claim text]
  Entities: [list of entities]
  LLM Comment: [Why this is fact-checkable]
  Source: image (ID: test-image-transcription)

Claim 2:
  ...
```

## Files

- `image_claim_extraction.py` - Main test script
- `images/` - Folder for test images (create and add images here)

## Notes

- The script uses **gpt-4o** model for both approaches
- Temperature is set to `0.0` for consistent results
- The transcription prompt emphasizes:
  - Visual formatting of text (bold, caps, colors)
  - Identification of famous people/politicians by name only
  - Unusual visual details relevant to fact-checking
- All claims are processed with the same validation and entity extraction

## Comparison Points

When comparing the two approaches, consider:

1. **Accuracy**: Which approach extracts more relevant claims?
2. **Context**: Does transcription help or lose important visual context?
3. **Quality**: Are the claims more normalized/self-contained in one approach?
4. **Entity Recognition**: Which approach identifies entities better?
5. **Cost**: Approach 2 makes two LLM calls instead of one

## Example Use Case

This script is useful for:
- Testing different prompt strategies for image claim extraction
- Evaluating whether image transcription improves claim quality
- Comparing direct vision capabilities vs. text-based processing
- Debugging claim extraction issues with specific images
