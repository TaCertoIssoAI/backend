
import re

VEREDICT_SUBSTR= "Veredito:"
JUSTIFICATION_SUBSTR = "Justificativa:"
CLAIM_SUBSTR = "Afirmação"

def replace_markdown_links(text: str) -> str:
    """
    replace markdown-style links [text](url) with just url.

    this is useful when the output platform doesn't support markdown formatting.
    removes the markdown syntax entirely, keeping only the URL.

    args:
        text: input text that may contain markdown links

    returns:
        text with all markdown links replaced by just the url

    example:
        >>> replace_markdown_links("Check [fifa.com](https://www.fifa.com/article) for more")
        'Check https://www.fifa.com/article for more'
        >>> replace_markdown_links("Multiple [link1](url1) and [link2](url2) here")
        'Multiple url1 and url2 here'
        >>> replace_markdown_links("([text](https://example.com))")
        '(https://example.com)'
    """
    # pattern matches: [any text](any url)
    # \[([^\]]+)\] matches [text] - group 1
    # \(([^\)]+)\) matches (url) - group 2
    # replace with just url (group 2), no extra parentheses
    pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
    return re.sub(pattern, r'\2', text)


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
