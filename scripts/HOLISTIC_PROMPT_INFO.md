# Holistic Vision-Based Claim Extraction Prompt

## Overview

A new local prompt has been added to the test script that explicitly instructs GPT-4o vision model to perform holistic image analysis for claim extraction.

## Location

File: `scripts/image_claim_extraction.py`  
Constant: `IMAGE_VISION_CLAIM_EXTRACTION_SYSTEM_PROMPT`

## Key Features

### 1. Explicit Image Input Instruction
The prompt explicitly tells the model it will receive an IMAGE as visual input, not just text.

### 2. Three-Step Analysis Process

**STEP 1 - Identify People and Characters:**
- Famous people (politicians, celebrities, historical figures)
- Fictional characters from movies, series, books, etc.
- Identifies by NAME only (no status, position, or alive/dead info)

**STEP 2 - Read All Text:**
- Reads all text in the image (titles, captions, headlines, quotes)
- Notes visual formatting (CAPS LOCK, bold, sizes, colors)

**STEP 3 - Holistic Interpretation:**
- CONNECTS visual elements + text + identified people/characters
- Interprets the COMPLETE MESSAGE the media is communicating
- Considers context (news, meme, cartoon, montage)
- Identifies the NARRATIVE about the real world

### 3. Holistic Claim Extraction

The prompt emphasizes extracting claims that result from the HOLISTIC interpretation:
- Not just text OR image separately
- But the COMBINATION of all elements
- The narrative/message that emerges from visual + text + context

### 4. Special Cases

**Memes, Cartoons, and Montages:**
- Identifies central message
- Uses contextual clues (text, recognizable people, symbols, event references)
- Extracts factual claims implied by the imagery
- Detects visual manipulation when relevant

### 5. Normalization Guidelines

- Self-contained claims (understandable without seeing the image)
- Replaces pronouns with specific names of identified people/entities
- Preserves critical context (dates, locations, specific numbers)
- Multiple claims from single image when applicable

## Example Comparisons

### Old Approach (Text-Based)
Image → OCR/Description → Text-based extraction
- Might miss visual context
- Separated analysis of text and visual elements

### New Approach (Holistic Vision)
Image → Identify people + Read text + Interpret holistically → Claims
- Politicians/celebrities identified by name
- Text read with formatting awareness
- Visual + text + people connected for complete message

## Example Outputs

**Example 1: Political Meme**
- Image: Politician X with overlaid text "Stole millions"
- Old: Might extract just "Someone stole millions"
- New: "Politician [specific name] stole millions" (combining visual ID + text)

**Example 2: Cartoon**
- Image: Cartoon showing government crushing workers
- Old: Might describe scene visually
- New: Extracts claim about government labor policies (if context/text indicates)

**Example 3: Celebrity Photo**
- Image: Photo of Celebrity Y with caption "Died yesterday"
- Old: "A person died yesterday"
- New: "Celebrity [specific name] died" (identifying person from photo)

## Usage in Script

The prompt is used in **Approach 1** of the test script:
- Function: `call_gpt4o_for_claim_extraction_with_image()`
- Model: GPT-4o with vision capabilities
- Temperature: 0.0 (for consistency)
- Output: Structured JSON with claims, entities, and LLM analysis

## Benefits

1. **Better Person Recognition**: Famous people and characters identified by name
2. **Context Awareness**: Understands memes, cartoons, and visual narratives
3. **Holistic Understanding**: Combines visual + text + context
4. **More Accurate Claims**: Claims reflect the complete message, not just parts
5. **Manipulation Detection**: Can identify when images appear edited/fake

## Testing

Run the script with test images to compare:
- **Approach 1**: Uses this holistic vision prompt directly
- **Approach 2**: Transcribes first, then extracts (separate steps)

```bash
python3 scripts/image_claim_extraction.py
```

Add test images to `scripts/images/` folder to see both approaches in action.
