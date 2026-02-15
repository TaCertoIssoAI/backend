"""
formats accumulated context entries into prompt sections ordered by reliability.

produces globally-numbered entries so the agent can reference [N] unambiguously.
"""

from __future__ import annotations

from app.models.agenticai import (
    FactCheckApiContext,
    GoogleSearchContext,
    WebScrapeContext,
)


def format_context(
    fact_check_results: list[FactCheckApiContext],
    search_results: dict[str, list[GoogleSearchContext]],
    scraped_pages: list[WebScrapeContext],
) -> str:
    """build the formatted context string with global numbering."""
    sections: list[str] = []
    counter = 1

    # === muito confiável ===
    muito_confiavel_lines: list[str] = []

    # fact-check API results
    if fact_check_results:
        muito_confiavel_lines.append("### Fact-Check API")
        for entry in fact_check_results:
            muito_confiavel_lines.append(
                f"[{counter}] Publisher: {entry.publisher} | Rating: {entry.rating}\n"
                f"    URL: {entry.url}\n"
                f"    Afirmação verificada: \"{entry.claim_text}\"\n"
                f"    Data da revisão: {entry.review_date or 'N/A'}"
            )
            counter += 1

    # domain-specific search results (muito confiável)
    for domain_key, label in [
        ("aosfatos", "Aos Fatos"),
        ("g1", "G1"),
        ("estadao", "Estadão"),
        ("folha", "Folha"),
    ]:
        entries = search_results.get(domain_key, [])
        if entries:
            muito_confiavel_lines.append(f"### Busca Web — {label}")
            for entry in entries:
                muito_confiavel_lines.append(
                    f"[{counter}] Title: \"{entry.title}\"\n"
                    f"    URL: {entry.url} | Domain: {entry.domain}\n"
                    f"    Snippet: \"{entry.snippet}\""
                )
                counter += 1

    if muito_confiavel_lines:
        sections.append(
            "## Fontes — Muito confiável\n\n" + "\n\n".join(muito_confiavel_lines)
        )

    # === neutro (general web search only) ===
    neutro_lines: list[str] = []

    geral_entries = search_results.get("geral", [])
    if geral_entries:
        neutro_lines.append("### Busca Web — Geral")
        for entry in geral_entries:
            neutro_lines.append(
                f"[{counter}] Title: \"{entry.title}\"\n"
                f"    URL: {entry.url} | Domain: {entry.domain}\n"
                f"    Snippet: \"{entry.snippet}\""
            )
            counter += 1

    if neutro_lines:
        sections.append("## Fontes — Neutro\n\n" + "\n\n".join(neutro_lines))

    # === pouco confiável (scraped pages) ===
    if scraped_pages:
        pouco_lines = ["### Conteúdo Extraído de Páginas"]
        for entry in scraped_pages:
            content_preview = entry.content[:500] if entry.content else "(vazio)"
            pouco_lines.append(
                f"[{counter}] Title: \"{entry.title}\" | URL: {entry.url}\n"
                f"    Status: {entry.extraction_status} | "
                f"Ferramenta: {entry.extraction_tool}\n"
                f"    Conteúdo (primeiros 500 chars): \"{content_preview}\""
            )
            counter += 1
        sections.append(
            "## Fontes — Pouco confiável\n\n" + "\n\n".join(pouco_lines)
        )

    if not sections:
        return "(nenhuma fonte coletada ainda)"

    return "\n\n".join(sections)
