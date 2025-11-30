
import re

VEREDICT_SUBSTR= "Veredito:"
JUSTIFICATION_SUBSTR = "Justificativa:"
CLAIM_SUBSTR = "Afirmação"

def remove_link_like_substrings(text: str) -> str:
    """
    Remove "LINK-like" substrings and URLs from a text.

    - Removes http/https URLs
    - Removes www.* URLs
    - Removes tokens starting with 'LINK' (e.g. LINK, LINK1, LINK_ABC)

    Returns a cleaned string with extra whitespace collapsed.
    """
    # Remove http(s) URLs
    text = re.sub(r'https?://\S+', '', text)

    # Remove www.* URLs
    text = re.sub(r'www\.\S+', '', text)

    # Remove LINK-like tokens (start with LINK and go until next whitespace)
    text = re.sub(r'\bLINK[^\s]*', '', text)

    # Collapse multiple spaces and strip ends
    text = re.sub(r'\s+', ' ', text).strip()

    return text
