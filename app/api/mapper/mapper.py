"""
mapper functions to convert between API models and internal data models.

this module provides functions to transform API request/response models
into the internal pipeline data structures and vice versa.
"""

import uuid
from typing import List, cast
from datetime import datetime

from app.models.api import Request, ContentType,AnalysisResponse
from app.models.commondata import DataSource
from app.models.factchecking import ClaimSourceType,FactCheckResult,VerdictTypeEnum
from app.observability.logger.logger import get_logger
from app.clients import get_analytics_url_for_fact_check

from .formating import VEREDICT_SUBSTR,JUSTIFICATION_SUBSTR,CLAIM_SUBSTR, remove_link_like_substrings, replace_markdown_links

logger = get_logger(__name__)


def map_content_type_to_source_type(content_type: ContentType) -> ClaimSourceType:
    """
    map API ContentType to internal ClaimSourceType.
    
    args:
        content_type: the content type from the API request
        
    returns:
        corresponding ClaimSourceType for internal use
    """
    mapping: dict[ContentType, ClaimSourceType] = {
        ContentType.TEXT: cast(ClaimSourceType, "original_text"),
        ContentType.IMAGE: cast(ClaimSourceType, "image"),
        ContentType.AUDIO: cast(ClaimSourceType, "audio_transcript"),
        ContentType.VIDEO: cast(ClaimSourceType, "video_transcript"),
    }
    
    return mapping[content_type]


def request_to_data_sources(
    request: Request,
    locale: str = "pt-BR"
) -> List[DataSource]:
    """
    convert an API Request into a list of DataSource objects.
    
    takes the unified request model from the API and transforms each
    ContentItem into a DataSource ready for the fact-checking pipeline.
    
    args:
        request: the API request containing content items
        locale: language locale for the data sources (default: pt-BR)
        
    returns:
        list of DataSource objects, one per content item
        
    example:
        >>> from app.models.api import Request, ContentItem, ContentType
        >>> request = Request(content=[
        ...     ContentItem(textContent="Neymar voltou ao Santos", type=ContentType.TEXT)
        ... ])
        >>> sources = request_to_data_sources(request)
        >>> len(sources)
        1
        >>> sources[0].source_type
        'original_text'
        >>> sources[0].original_text
        'Neymar voltou ao Santos'
    """
    data_sources: List[DataSource] = []
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    for content_item in request.content:
        # generate unique ID for this data source
        source_id = f"{uuid.uuid4()}"
        
        # map content type to source type
        source_type = map_content_type_to_source_type(content_item.type)
        
        # create DataSource
        data_source = DataSource(
            id=source_id,
            source_type=source_type,
            original_text=content_item.textContent,
            locale=locale,
            timestamp=timestamp,
        )
        
        data_sources.append(data_source)
    
    return data_sources


def _get_verdict_summary(all_verdicts: list) -> str:
    """
    generate a summary of verdicts showing how many claims were classified as each type.

    args:
        all_verdicts: list of ClaimVerdict objects

    returns:
        formatted string like "3 de 5 afirmações foram classificadas como Verdadeiras"

    example:
        >>> verdicts = [ClaimVerdict(..., verdict="Verdadeiro"), ClaimVerdict(..., verdict="Falso")]
        >>> _get_verdict_summary(verdicts)
        "1 de 2 afirmações foram classificadas como Verdadeiras"
    """
    if not all_verdicts:
        return ""

    total_claims = len(all_verdicts)

    # count verdicts by type
    verdadeiro_count = sum(1 for v in all_verdicts if v.verdict == VerdictTypeEnum.VERDADEIRO)
    falso_count = sum(1 for v in all_verdicts if v.verdict == VerdictTypeEnum.FALSO)
    fontes_insuficientes_count = sum(1 for v in all_verdicts if v.verdict == VerdictTypeEnum.FONTES_INSUFICIENTES)
    fora_de_contexto_count = sum(1 for v in all_verdicts if v.verdict == VerdictTypeEnum.FORA_DE_CONTEXTO)

    # check edge cases where all claims have the same verdict
    if verdadeiro_count == total_claims:
        return "Todas as afirmações foram classificadas como verdadeiras."

    if falso_count == total_claims:
        return "Todas as afirmações foram classificadas como Falsas."

    if fontes_insuficientes_count == total_claims:
        return "Não foi possível verificar nenhuma das informações."
    
    if fora_de_contexto_count == total_claims:
        return "Todas as afirmações foram classificadas como fora de contexto."

    # default case: show count of true claims
    verdadeiro_count = max(0, verdadeiro_count)
    return f"{verdadeiro_count} de {total_claims} afirmações foram classificadas como Verdadeiras."


def fact_check_result_to_response(
    msg_id: str,
    result: FactCheckResult,
    fact_check_results: list | None = None,
    search_results: dict | None = None,
    scraped_pages: list | None = None,
) -> AnalysisResponse:
        all_verdicts = []
        for ds_result in result.results:
            all_verdicts.extend(ds_result.claim_verdicts)

        # build rationale text from verdicts
        if all_verdicts:
            rationale_parts = []

            # add overall summary first if present
            if result.overall_summary:
                # add "Resumo Geral" header
                rationale_parts.append("*Resumo Geral*:\n")

                # add verdict count summary
                verdict_summary = _get_verdict_summary(all_verdicts)
                if verdict_summary:
                    rationale_parts.append(verdict_summary)

                # add the actual overall summary text
                rationale_parts.append(f"\n{result.overall_summary}\n")

            analytics_url = get_analytics_url_for_fact_check(msg_id)
            rationale_parts.append("Saiba mais sobre esse julgamento no nosso website:")
            rationale_parts.append(f"{analytics_url}\n")

            # add claims section header
            rationale_parts.append("\n*Análise por afirmação*:")

            # add each verdict with its justification
            for i, verdict_item in enumerate(all_verdicts, 1):
                rationale_parts.append(f"\n*{CLAIM_SUBSTR} {i}*: {verdict_item.claim_text}")
                rationale_parts.append(f"*{VEREDICT_SUBSTR}* {verdict_item.verdict}")
                rationale_parts.append(f"*{JUSTIFICATION_SUBSTR}* {verdict_item.justification}")

            rationale = "\n".join(rationale_parts)

            # build citation section using the same numbering as format_context
            citation_text = _build_citations_from_source_refs(
                rationale,
                result,
                fact_check_results or [],
                search_results or {},
                scraped_pages or [],
            )
            if citation_text:
                rationale = rationale + citation_text
        else:
            rationale = "Nenhuma alegação verificável foi encontrada no conteúdo fornecido.\n"
            if result.overall_summary:
                rationale = rationale + result.overall_summary

        #this is for the openAI model output
        rationale = replace_markdown_links(rationale)

        resp_without_links = remove_link_like_substrings(rationale)
        return AnalysisResponse(
            message_id=msg_id,
            rationale=rationale,
            responseWithoutLinks=resp_without_links,
        )


def _build_citations_from_source_refs(
    rationale_text: str,
    result: FactCheckResult,
    fact_check_results: list,
    search_results: dict,
    scraped_pages: list,
) -> str:
    """build the citation section using the same numbering as format_context.

    uses build_source_reference_list to create [N] → (title, url) mapping
    that matches the numbering the LLM used during adjudication, then
    filter_cited_references to keep only those actually cited in text.
    """
    from app.agentic_ai.prompts.context_formatter import (
        build_source_reference_list,
        filter_cited_references,
    )

    source_refs = build_source_reference_list(
        fact_check_results, search_results, scraped_pages,
    )

    if not source_refs:
        return ""

    # collect all texts where citations may appear
    cited_texts = [result.overall_summary or ""]
    for ds_result in result.results:
        for cv in ds_result.claim_verdicts:
            cited_texts.append(cv.justification)

    cited_refs = filter_cited_references(source_refs, *cited_texts)

    if not cited_refs:
        return ""

    citation_parts = ["\n\n*Fontes*:"]
    for num, title, url in cited_refs:
        citation_parts.append(f"\n[{num}] {title}")
        citation_parts.append(f"    URL: {url}")

    return "\n".join(citation_parts)