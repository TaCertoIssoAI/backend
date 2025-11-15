"""
Claim Extraction Module - Step 2 of Fact-Checking Pipeline

This module implements claim extraction using OpenAI LLM with structured output mode.
Takes raw user text and extracts individual verifiable claims with associated metadata.

Following LangChain best practices:
- Structured outputs with Pydantic models
- LCEL composition
- Async-first design
- Stateless chains
- Portuguese (pt-BR) language support
"""

from typing import List
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException

from app.models.factchecking import (
    UserInput,
    ClaimExtractionOutput
)
from app.core.config import get_settings

settings = get_settings()


class ClaimExtractor:
    """
    Extracts verifiable claims from raw user text using OpenAI LLM.

    Follows LangChain best practices:
    - Structured output with with_structured_output()
    - LCEL chain composition
    - Stateless design
    - Type-safe Pydantic models
    """

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.0):
        """Initialize the claim extractor with OpenAI model."""

        # Get fresh settings to ensure .env is loaded
        current_settings = get_settings()
        
        # Initialize OpenAI model following LangChain best practices
        self.model = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=current_settings.OPENAI_API_KEY
        )

        # Create prompt template following consistent message handling
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("user", self._get_user_prompt())
        ])

        # Create LCEL chain with structured output
        self.chain = (
            self.prompt
            | self.model.with_structured_output(ClaimExtractionOutput)
        )

    def _get_system_prompt(self) -> str:
        """
        System prompt for claim extraction in Portuguese.
        Follows LangChain best practice of keeping prompts in separate methods.
        """
        return """Você é um especialista em extração de alegações. Sua tarefa é analisar textos de usuários do WhatsApp em português brasileiro e extrair TODAS as alegações factuais presentes, independentemente de serem verdadeiras, falsas, controversas ou especulativas.

REGRAS IMPORTANTES:
1. Extraia TODAS as alegações factuais - não filtre por veracidade ou verificabilidade
2. Inclua alegações controversas, especulativas ou sensíveis - elas serão fact-checked posteriormente com fontes confiáveis
3. Se houver múltiplas alegações, extraia cada uma separadamente
4. Identifique e extraia qualquer URL/link presente no texto original
5. Para cada alegação, forneça um comentário analítico neutro sobre seu conteúdo
6. Inclua declarações sobre eventos, pessoas, organizações, datas, números, políticas, relações, características
7. Normalize o texto das alegações (remova gírias, corrija erros óbvios, mantenha o sentido)

CRITICAL - HANDLING VERIFICATION REQUESTS AND LINKS:
8. EXTRAIA APENAS o que o usuário EXPLICITAMENTE declarou no texto
9. EXCEÇÃO - Se o usuário pedir para "verificar", "checar", "confirmar" ou similar, OU compartilhar apenas um link, extraia as principais alegações do conteúdo/título da URL
10. Para pedidos de verificação ou links isolados, identifique 1-3 alegações principais baseadas no título/conteúdo
11. NÃO faça suposições quando há alegações explícitas + links - mantenha apenas as alegações do usuário

NOTA IMPORTANTE: Nossa pipeline fará fact-checking posterior usando fontes confiáveis e especializadas. Sua função é apenas extrair, não filtrar por veracidade.

EXEMPLOS DE ALEGAÇÕES PARA EXTRAIR:
✓ "O governo anunciou novas políticas ontem"
✓ "A vacina X causa infertilidade"
✓ "Pessoas com olhos azuis são mais inteligentes"
✓ "Político X tem relacionamento com Político Y"
✓ "O TSE proibiu pesquisas eleitorais em 2024"
✓ "A empresa Y demitiu 1000 funcionários"

EXEMPLOS COM URLS:

CASO 1 - Alegação explícita + link:
Texto: "Flavio bolsonaro amo o PT segundo esse link: https://site.com/flavio-defende-anistia"
✓ CORRETO: Extrair "Flavio Bolsonaro ama o PT"
✗ ERRADO: Extrair "Flavio Bolsonaro defende anistia" (baseado na URL)

CASO 2 - Pedido de verificação:
Texto: "Verifique essa reportagem: https://g1.com/brasil-perdeu-15-por-cento-praias"
✓ CORRETO: Extrair "Brasil perdeu 15% das praias" (do título/conteúdo)
✓ CORRETO: Extrair alegações principais do artigo

CASO 3 - Link isolado:
Texto: "https://site.com/vacinas-causam-autismo"
✓ CORRETO: Extrair "Vacinas causam autismo" (do título/conteúdo)

EXEMPLOS DO QUE NÃO EXTRAIR:
✗ "O que você acha sobre...?"
✗ Alegações quando há texto explícito + link (usar só o texto explícito)

Responda sempre em português brasileiro."""

    def _get_user_prompt(self) -> str:
        """
        User prompt template for claim extraction.
        Uses parameterized input following LangChain best practices.
        """
        return """Texto do usuário: "{text}"

Contexto adicional (se disponível): {context}

Analise o texto acima e extraia todas as alegações verificáveis. Para cada alegação, forneça:
1. O texto normalizado da alegação
2. Qualquer link/URL encontrado no texto original
3. Seu comentário sobre a verificabilidade desta alegação
4. Entidades relevantes (pessoas, organizações, lugares, datas)

Se não houver alegações verificáveis, retorne uma lista vazia mas explique o motivo nas notas de processamento."""

    def _extract_urls_from_text(self, text: str) -> List[str]:
        """
        Extract URLs from text using regex.
        Helper method following separation of concerns principle.
        """
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        return urls

    async def extract_claims(self, user_input: UserInput) -> ClaimExtractionOutput:
        """
        Extract claims from user input using async LLM call.

        Args:
            user_input: UserInput model with text and metadata

        Returns:
            ClaimExtractionOutput: Structured output with extracted claims

        Raises:
            OutputParserException: If LLM output doesn't match expected schema
            Exception: For other processing errors
        """
        try:
            # Extract URLs from original text for inclusion in claims
            extracted_urls = self._extract_urls_from_text(user_input.text)

            # Prepare input for the chain following stateless design
            chain_input = {
                "text": user_input.text,
                "context": user_input.context or "Nenhum contexto adicional fornecido."
            }

            # Call LLM chain using async invoke following best practices
            result = await self.chain.ainvoke(chain_input)

            # Post-process to ensure URLs are included in claims
            for claim in result.claims:
                if not claim.links and extracted_urls:
                    # Add URLs to claims that don't have any
                    claim.links = extracted_urls

            # Validate that we have meaningful output
            if not result.claims:
                result.processing_notes = (
                    "Nenhuma alegação verificável encontrada no texto. "
                    "O texto pode conter apenas perguntas, opiniões ou especulações."
                )

            return result

        except OutputParserException as e:
            # Handle structured output parsing errors
            fallback_result = ClaimExtractionOutput(
                original_text=user_input.text,
                claims=[],
                processing_notes=f"Erro ao processar resposta do LLM: {str(e)}"
            )
            return fallback_result

        except Exception as e:
            # Handle other errors gracefully
            fallback_result = ClaimExtractionOutput(
                original_text=user_input.text,
                claims=[],
                processing_notes=f"Erro durante extração de alegações: {str(e)}"
            )
            return fallback_result


# Factory function following LangChain best practices
def create_claim_extractor(model_name: str = "gpt-4o") -> ClaimExtractor:
    """
    Factory function to create a ClaimExtractor instance.

    Args:
        model_name: OpenAI model name to use

    Returns:
        Configured ClaimExtractor instance
    """
    return ClaimExtractor(model_name=model_name)


# Async helper function for direct usage
async def extract_claims_from_text(text: str) -> ClaimExtractionOutput:
    """
    Convenience function to extract claims from raw text.

    Args:
        text: Raw text to analyze

    Returns:
        ClaimExtractionOutput with extracted claims
    """
    extractor = create_claim_extractor()

    user_input = UserInput(
        text=text,
        locale="pt-BR"
    )

    return await extractor.extract_claims(user_input)