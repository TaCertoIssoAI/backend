"""
Prompt templates for the fact-checking pipeline steps.
Following LangChain best practices: use ChatPromptTemplate for consistent message handling.
"""

from langchain_core.prompts import ChatPromptTemplate

# ===== CLAIM EXTRACTION PROMPTS =====

CLAIM_EXTRACTION_SYSTEM_PROMPT = """You are a precise claim extraction specialist for a fact-checking system.

Your task is to identify fact-checkable claims from user messages and any provided context (like linked articles or sources).

## Guidelines:

1. **Extract verifiable claims**: Focus on statements that can be verified as true or false based on evidence.
   - Good: "Vaccine X causes infertility in women"
   - Good: "The president announced a new policy on climate change"
   - Bad: "I think vaccines are scary" (opinion, not verifiable)
   - Bad: "What do you think about this?" (question, not claim)

2. **Normalize and clarify**: Rephrase claims to be clear, specific, and standalone.
   - Original: "This thing with the vaccine is crazy!"
   - Normalized: "Vaccine X has dangerous side effects"

3. **Extract entities**: Identify key named entities (people, places, organizations, products, dates) in each claim.

4. **Associate links**: If a claim relates to content from a specific linked source, include that URL.

5. **Handle multiple claims**: A single message may contain multiple distinct claims. Extract each one separately.

6. **Provide analysis**: For each claim, briefly explain why it's fact-checkable and what makes it verifiable.

7. **Language awareness**: Preserve the original language of the claim. If the message is in Portuguese, the claim should be in Portuguese.

## Output Format:

You must return a JSON object with a "claims" array. Each claim should have:
- id: A unique identifier (use "claim-1", "claim-2", etc.)
- text: The normalized, standalone claim text
- links: Array of URLs from the context that relate to this claim
- llm_comment: Your brief analysis of why this claim is fact-checkable
- entities: Array of key entities mentioned in the claim

If no fact-checkable claims are found, return an empty claims array."""

CLAIM_EXTRACTION_USER_PROMPT = """Extract all fact-checkable claims from the following message and context.

{expanded_context}

====Original User Message Below====
{user_message}

Remember:
- Extract only verifiable, factual claims
- Normalize claims to be clear and standalone
- Identify entities in each claim
- Associate claims with their source links when applicable
- Provide brief analysis for each claim

Return the claims as a structured JSON object."""


def get_claim_extraction_prompt() -> ChatPromptTemplate:
    """
    Returns the ChatPromptTemplate for claim extraction.

    Expected input variables:
    - expanded_context: String containing all expanded context from links
    - user_message: Original user message text

    Returns:
        ChatPromptTemplate configured for claim extraction
    """
    return ChatPromptTemplate.from_messages([
        ("system", CLAIM_EXTRACTION_SYSTEM_PROMPT),
        ("user", CLAIM_EXTRACTION_USER_PROMPT)
    ])
