"""
ID generation utilities for the application.

Uses NanoID for generating short, URL-safe unique identifiers.
"""

from nanoid import generate

_ALPHABET ="0123456789abcdefghijklmnopqrst"

def generate_message_id() -> str:
    """
    generate a unique message ID using NanoID.

    returns:
        12-character URL-safe unique identifier

    example:
        >>> msg_id = generate_message_id()
        >>> len(msg_id)
        12
    """
    return generate(size=12)


def generate_claim_id() -> str:
    """
    generate a unique claim ID using NanoID.

    returns:
        12-character URL-safe unique identifier

    example:
        >>> claim_id = generate_claim_id()
        >>> len(claim_id)
        12
    """
    return generate(size=12)


def generate_source_id() -> str:
    """
    generate a unique source ID using NanoID.

    returns:
        12-character URL-safe unique identifier

    example:
        >>> source_id = generate_source_id()
        >>> len(source_id)
        12
    """
    return generate(size=12)


def generate_link_id() -> str:
    """
    generate a unique link ID using NanoID.

    returns:
        12-character URL-safe unique identifier

    example:
        >>> link_id = generate_link_id()
        >>> len(link_id)
        12
    """
    return generate(size=12)


def generate_document_id() -> str:
    """
    generate a unique document ID using NanoID.

    returns:
        12-character URL-safe unique identifier

    example:
        >>> doc_id = generate_document_id()
        >>> len(doc_id)
        12
    """
    return generate(size=12)


def generate_task_id() -> str:
    """
    generate a unique task ID using NanoID.

    returns:
        12-character URL-safe unique identifier

    example:
        >>> task_id = generate_task_id()
        >>> len(task_id)
        12
    """
    return generate(size=12)
