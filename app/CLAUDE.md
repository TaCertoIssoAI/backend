# Fact Checking Pipeline - Architecture and Data Model

This document explains the fact checking pipeline using:

- The architecture diagram (multi modal intake, claim extraction, tools, final LLM)
- The Pydantic schema that defines the data contracts between each step

It is meant as a deployment friendly, implementation ready description of how data flows from the user message to the final fact check answer and analytics.

---

## 1. High Level Architecture

The architecture diagram shows the following flow:

1. **Channel intake and multi modal preprocessing**

   - Input from WhatsApp ("ZAP"), web, etc.
   - User sends text, links, images, audio, video.
   - A preprocessor outside the LLM pipeline turns all of that into:
     - Canonical text (including link URLs)
     - Metadata like timestamp, locale, ids

2. **LLM pipeline (text only)**

   Once we have canonical text, the core pipeline runs:

   1. **User input step**  
      - Defines the raw message as seen by the pipeline.
   2. **Context expansion**  
      - Extracts and enriches links and other inline sources.
      - Builds an "expanded context" that gives more info about the message content.
   3. **Claim extraction (LLM)**  
      - Identifies fact checkable claims in the message and normalizes them.
   4. **Evidence gathering (multi step)**  
      - For each claim:
        - Hit fact checking APIs
        - Do web searches
        - Call internal tools (via MCP or other APIs)
      - Aggregate all evidence per claim.
   5. **Final adjudication (LLM)**  
      - A strong model looks at:
        - Original message
        - Structured claims
        - Evidence per claim
      - Produces a verdict and explanation in user friendly text.

3. **Output and analytics**

   - The answer is sent back to the user (for example through WhatsApp).
   - A rich analytics object is stored with:
     - All step outputs
     - All citations and links
     - Engineering timings and token usage

---

## 2. Cross Cutting Data and Engineering Analytics

Two helper models travel across many steps and are not tied to a single stage.

### 2.1 CommonPipelineData

```python
class CommonPipelineData(BaseModel):
    """Common data that is crucial for the fact-checking pipeline but is used at several different steps"""
    message_id: str = Field(..., description="Internal id for the request")
    message_text: str = Field(..., description="Original Message text")
    
    locale: str = Field(default="pt-BR", description="Language locale")
    timestamp: Optional[str] = Field(None, description="When the message was sent")
```