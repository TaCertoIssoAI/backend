"""
utilities for removing brazilian personal identifiable information (PII).

removes CPF, CNPJ, CEP, and credit card numbers from text to protect user privacy.

references:
- CPF/CNPJ/CEP regex patterns: https://gist.github.com/jrrio/b2f80695c2cc076ca4727fb32e94a447
- credit card patterns: https://www.regular-expressions.info/creditcard.html
"""

import re
from typing import List
from app.models.api import Request, AnalysisResponse, ContentItem


# regex patterns for brazilian IDs

# CPF pattern - matches both formatted (XXX.XXX.XXX-XX) and unformatted (11 digits)
CPF_PATTERN = re.compile(
    r'\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b'
)

# CNPJ pattern - matches both formatted (XX.XXX.XXX/XXXX-XX) and unformatted (14 digits)
CNPJ_PATTERN = re.compile(
    r'\b\d{2}\.?\d{3}\.?\d{3}/?-?\d{4}-?\d{2}\b'
)

# CEP pattern - matches both formatted (XXXXX-XXX) and unformatted (8 digits)
CEP_PATTERN = re.compile(
    r'\b\d{5}-?\d{3}\b'
)

# credit card pattern - matches 13-19 digit sequences with optional spaces/dashes
# covers visa, mastercard, amex, diners, discover, etc.
CREDIT_CARD_PATTERN = re.compile(
    r'\b(?:\d[ -]*?){13,19}\b'
)

# more specific credit card patterns for better detection
VISA_PATTERN = re.compile(
    r'\b4\d{3}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}(?:\d{3})?\b'
)

MASTERCARD_PATTERN = re.compile(
    r'\b(?:5[1-5]\d{2}|2(?:22[1-9]|2[3-9]\d|[3-6]\d{2}|7[01]\d|720))[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b'
)

AMEX_PATTERN = re.compile(
    r'\b3[47]\d{2}[ -]?\d{6}[ -]?\d{5}\b'
)

DINERS_PATTERN = re.compile(
    r'\b3(?:0[0-5]|[68]\d)\d[ -]?\d{6}[ -]?\d{4}\b'
)


def remove_cpf(text: str, replacement: str = "[CPF REMOVIDO]") -> str:
    """
    remove CPF numbers from text.

    args:
        text: input text that may contain CPF
        replacement: string to replace CPF with

    returns:
        text with CPF numbers replaced
    """
    return CPF_PATTERN.sub(replacement, text)


def remove_cnpj(text: str, replacement: str = "[CNPJ REMOVIDO]") -> str:
    """
    remove CNPJ numbers from text.

    args:
        text: input text that may contain CNPJ
        replacement: string to replace CNPJ with

    returns:
        text with CNPJ numbers replaced
    """
    return CNPJ_PATTERN.sub(replacement, text)


def remove_cep(text: str, replacement: str = "[CEP REMOVIDO]") -> str:
    """
    remove CEP numbers from text.

    args:
        text: input text that may contain CEP
        replacement: string to replace CEP with

    returns:
        text with CEP numbers replaced
    """
    return CEP_PATTERN.sub(replacement, text)


def remove_credit_cards(text: str, replacement: str = "[CARTÃO REMOVIDO]") -> str:
    """
    remove credit card numbers from text.

    uses multiple patterns to detect visa, mastercard, amex, diners, etc.

    args:
        text: input text that may contain credit card numbers
        replacement: string to replace card numbers with

    returns:
        text with credit card numbers replaced
    """
    # apply specific patterns first for better accuracy
    text = VISA_PATTERN.sub(replacement, text)
    text = MASTERCARD_PATTERN.sub(replacement, text)
    text = AMEX_PATTERN.sub(replacement, text)
    text = DINERS_PATTERN.sub(replacement, text)

    # apply general pattern as fallback
    text = CREDIT_CARD_PATTERN.sub(replacement, text)

    return text


def remove_all_pii(text: str) -> str:
    """
    remove all brazilian PII from text.

    applies all removal functions in sequence:
    - CPF numbers
    - CNPJ numbers
    - CEP numbers
    - credit card numbers

    args:
        text: input text that may contain PII

    returns:
        text with all PII removed
    """
    if not text:
        return text

    # remove in order of specificity
    # (more specific patterns first to avoid conflicts)
    text = remove_credit_cards(text)
    text = remove_cnpj(text)  # before CPF as CNPJ is longer
    text = remove_cpf(text)
    text = remove_cep(text)

    return text


def sanitize_request(request: Request) -> Request:
    """
    remove PII from all content items in a request.

    creates a new request with sanitized content.

    args:
        request: original request with potentially sensitive data

    returns:
        new request with PII removed from all text content
    """
    sanitized_content: List[ContentItem] = []

    for item in request.content:
        # create new content item with sanitized text
        sanitized_text = ""
        if item.textContent:
            sanitized_text = remove_all_pii(item.textContent)

        sanitized_item = ContentItem(
            textContent=sanitized_text,
            type=item.type
        )
        sanitized_content.append(sanitized_item)

    # create new request with sanitized content
    sanitized_request = Request(content=sanitized_content)

    return sanitized_request


def sanitize_response(response: AnalysisResponse) -> AnalysisResponse:
    """
    remove PII from response fields.

    creates a new response with sanitized rationale and responseWithoutLinks.

    args:
        response: original response with potentially sensitive data

    returns:
        new response with PII removed from text fields
    """
    sanitized_rationale = remove_all_pii(response.rationale) if response.rationale else response.rationale
    sanitized_response_without_links = remove_all_pii(response.responseWithoutLinks) if response.responseWithoutLinks else response.responseWithoutLinks

    # create new response with sanitized fields
    sanitized_response = AnalysisResponse(
        message_id=response.message_id,
        rationale=sanitized_rationale,
        responseWithoutLinks=sanitized_response_without_links,
    )

    return sanitized_response
