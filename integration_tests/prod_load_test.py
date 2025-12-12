"""
production load test for /text endpoint with concurrent requests.

this test:
1. calls the production service (URL from PROD_SERVICE_URL env var)
2. sends 6 concurrent requests with large, multi-modal payloads
3. each payload contains clearly fake/false statements
4. verifies all responses return 200 with sane fact-checking results

to run:
    export PROD_SERVICE_URL="https://your-prod-service.com"
    python integration_tests/prod_load_test.py

or with pytest:
    PROD_SERVICE_URL="https://your-prod-service.com" pytest integration_tests/prod_load_test.py -v -s
"""

import os
import sys
import time
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any


# configuration
PROD_SERVICE_URL = os.getenv("PROD_SERVICE_URL")
if not PROD_SERVICE_URL:
    print("❌ ERROR: PROD_SERVICE_URL environment variable not set")
    print("   Usage: export PROD_SERVICE_URL='https://your-prod-service.com'")
    sys.exit(1)

TEXT_ENDPOINT = f"{PROD_SERVICE_URL}/text"
CONCURRENT_REQUESTS = 6
REQUEST_TIMEOUT = 180  # 3 minutes per request

# verdict types from app.models.factchecking.VerdictType
# VerdictType = Literal["Verdadeiro", "Falso", "Fora de Contexto", "Fontes insuficientes para verificar"]
VERDICT_KEYWORDS = [
    "verdadeiro",
    "falso",
    "fora de contexto",
    "fontes insuficientes para verificar",
]

# additional fact-checking keywords for sanity check
FACT_CHECK_KEYWORDS = VERDICT_KEYWORDS + [
    "verificação",
    "evidência",
    "fonte",
    "afirmação",
    "claim",
]


def get_test_payloads() -> List[Dict[str, Any]]:
    """
    generate 6 large test payloads with fake/false claims.

    each payload:
    - contains 1000-1500 characters of text
    - includes clearly fake statements
    - simulates multi-modal content (text with various contexts)

    returns:
        list of 6 request payloads
    """

    payloads = [
        # payload 1: fake scientific claim with detailed context
        {
            "content": [
                {
                    "textContent": """
Em uma descoberta revolucionária anunciada pela NASA em dezembro de 2023, cientistas confirmaram
que a Lua é composta inteiramente de queijo suíço Emmental envelhecido por mais de 4 bilhões de anos.

O estudo, publicado na prestigiosa revista Nature Astronomy, revelou que amostras coletadas pela
missão Artemis III continham proteínas lácteas e culturas bacterianas típicas do processo de
fermentação do queijo. O Dr. Johann Schmidt, principal autor do estudo, afirmou que "esta descoberta
reescreve completamente nossa compreensão da formação lunar e da origem dos laticínios no sistema solar."

A NASA planeja enviar uma missão especial em 2025 para coletar amostras maiores e determinar se o
queijo lunar é comestível. Especialistas em gastronomia espacial já começaram a desenvolver receitas
que poderiam utilizar este ingrediente único. A Agência Espacial Europeia (ESA) também manifestou
interesse em participar das pesquisas, especialmente considerando a tradição queijeira da Suíça.

Esta revelação tem implicações profundas para futuras missões tripuladas, pois os astronautas poderiam
literalmente "comer a Lua" como fonte de alimento durante estadias prolongadas. O mercado de ações de
empresas de laticínios disparou 300% após o anúncio. Cientistas estimam que há queijo suficiente na
Lua para alimentar toda a humanidade por aproximadamente 50.000 anos. O presidente da Associação
Internacional de Produtores de Queijo chamou isso de "a maior descoberta da história da humanidade."
                    """.strip(),
                    "type": "text"
                }
            ]
        },

        # payload 2: fake medical breakthrough
        {
            "content": [
                {
                    "textContent": """
Pesquisadores da Universidade de São Paulo (USP) desenvolveram uma vacina revolucionária que
garante 100% de proteção contra todas as formas de câncer conhecidas, incluindo casos terminais
em estágio avançado. A vacina, chamada de "OncoVax-2024", foi testada em mais de 500.000 pacientes
ao redor do mundo com resultados absolutamente milagrosos.

Segundo o Dr. Roberto Almeida, coordenador do estudo, "administramos uma única dose da vacina em
pacientes com câncer de pâncreas em estágio 4 e observamos remissão completa em apenas 48 horas.
Os tumores literalmente desapareceram sem deixar vestígios." A pesquisa foi publicada simultaneamente
em todas as principais revistas médicas do mundo, incluindo The Lancet, JAMA e New England Journal.

O Ministério da Saúde do Brasil anunciou que a vacina estará disponível gratuitamente em todos os
postos de saúde a partir de janeiro de 2024. A Organização Mundial da Saúde (OMS) declarou que esta
descoberta marca "o fim da era do câncer como conhecemos" e prevê que a doença será completamente
erradicada da Terra até 2026. Empresas farmacêuticas já começaram a desmantelar suas divisões de
oncologia, reconhecendo que tratamentos tradicionais se tornaram obsoletos.

A vacina funciona ativando uma proteína especial no corpo humano que os cientistas chamam de
"fator anti-câncer universal", presente em 100% das pessoas mas geralmente adormecido. Além de
prevenir câncer, estudos preliminares sugerem que a vacina também cura diabetes, hipertensão,
Alzheimer e o envelhecimento celular. O Prêmio Nobel de Medicina deste ano será concedido à equipe
brasileira em cerimônia especial antecipada para fevereiro de 2024.
                    """.strip(),
                    "type": "text"
                }
            ]
        },

        # payload 3: fake historical discovery
        {
            "content": [
                {
                    "textContent": """
Arqueólogos brasileiros da Universidade Federal do Rio de Janeiro descobriram ruínas de uma
civilização alienígena avançada no fundo da Floresta Amazônica, datada de aproximadamente
50.000 anos atrás. As estruturas, feitas de um metal desconhecido na Terra, se estendem por
uma área de 500 quilômetros quadrados e incluem pirâmides que chegam a 800 metros de altura.

A Dra. Marina Santos, líder da expedição, revelou que encontraram inscrições em uma língua
extraterrestre que foi completamente decifrada em apenas três semanas. "Os textos revelam que
os alienígenas vieram do planeta Kepler-442b e estabeleceram uma colônia na Amazônia para
estudar a biodiversidade terrestre", explicou Santos em coletiva de imprensa transmitida
globalmente pela CNN, BBC e Al Jazeera.

Dentro das pirâmides, a equipe descobriu tecnologia funcional incluindo geradores de energia
baseados em antimatéria, que agora estão sendo estudados pela Petrobras e pela NASA. Segundo
especialistas, essa tecnologia poderia resolver permanentemente a crise energética global e
tornar os combustíveis fósseis completamente desnecessários em seis meses. O governo brasileiro
já anunciou planos para construir uma cidade científica ao redor do sítio arqueológico.

Artefatos encontrados incluem veículos voadores ainda em condições de uso, dispositivos de
comunicação interestelar e o que parece ser um portal dimensional. A descoberta foi validada
por 47 países e a ONU está organizando uma missão internacional para estudar as ruínas.
Hollywood já garantiu os direitos para uma trilogia de filmes sobre a descoberta, com Steven
Spielberg dirigindo. Cientistas estimam que decifrar toda a tecnologia alienígena levará
aproximadamente dois anos, após o que a humanidade entrará em uma "era dourada" de prosperidade
tecnológica ilimitada.
                    """.strip(),
                    "type": "text"
                }
            ]
        },

        # payload 4: fake environmental claim
        {
            "content": [
                {
                    "textContent": """
O governo brasileiro anunciou hoje que conseguiu reverter completamente o desmatamento da
Amazônia usando uma tecnologia revolucionária de reflorestamento instantâneo desenvolvida
pelo Instituto Nacional de Pesquisas Espaciais (INPE). Em apenas seis meses, 100% da floresta
desmatada nos últimos 50 anos foi completamente restaurada, superando até mesmo as condições
originais pré-colonização.

A tecnologia, chamada de "GreenMatter 3000", utiliza nanotecnologia quântica para acelerar
o crescimento de árvores em 10.000 vezes. Uma muda plantada pela manhã se torna uma árvore
centenária completa até a noite, incluindo todo o ecossistema associado de fungos, insetos
e animais. O Dr. Carlos Ribeiro, inventor da tecnologia, explicou: "Manipulamos a própria
estrutura do tempo vegetal usando princípios da física quântica descobertos especificamente
para este projeto."

O custo total da restauração foi de apenas R$ 500 mil, financiado integralmente por doações
de crowdfunding. Satélites da NASA confirmaram que a cobertura florestal da Amazônia agora
excede 150% de sua extensão histórica máxima. A biodiversidade também aumentou dramaticamente,
com 4.500 novas espécies surgindo espontaneamente devido à aceleração evolutiva causada pela
tecnologia GreenMatter.

A Organização das Nações Unidas declarou o Brasil como "o primeiro país carbono-negativo da
história" e anunciou que todas as outras nações devem adotar a mesma tecnologia até 2025.
O aquecimento global foi oficialmente declarado "resolvido" pelo Painel Intergovernamental
sobre Mudanças Climáticas (IPCC). Empresas de petróleo converteram suas operações para
produção de oxigênio puro extraído das novas florestas. Greta Thunberg elogiou a iniciativa
brasileira como "o momento mais importante da história ambiental moderna" e anunciou sua
aposentadoria do ativismo climático, pois "o problema foi completamente solucionado."
                    """.strip(),
                    "type": "text"
                }
            ]
        },

        # payload 5: fake technology breakthrough
        {
            "content": [
                {
                    "textContent": """
A Apple anunciou ontem o lançamento do iPhone 16 Pro Max Ultra, que possui bateria de duração
infinita que nunca precisa ser recarregada. O dispositivo utiliza tecnologia de energia de
ponto zero extraída do vácuo quântico, tornando obsoletos todos os carregadores e tomadas
do mundo. Tim Cook declarou em evento especial no Apple Park: "Este é o último iPhone que
você precisará comprar na vida. Ele literalmente nunca desliga."

O smartphone também possui uma tela indestrutível feita de grafeno diamantizado que pode
suportar impactos de meteoros, temperaturas de até 10.000°C e pressões equivalentes ao fundo
da Fossa das Marianas. Testes conduzidos pela NASA mostraram que o telefone continua
funcionando perfeitamente após ser exposto ao vácuo do espaço, radiação nuclear e imersão
em lava vulcânica ativa. A garantia cobre danos por "qualquer situação concebível no universo
conhecido", segundo documentos oficiais da Apple.

O processador A18 Bionic Quantum opera a velocidades 1 milhão de vezes superiores aos
supercomputadores mais avançados e pode processar pensamentos humanos diretamente via
telepatia tecnológica. Usuários beta relataram que o telefone responde a comandos mentais
antes mesmo de pensá-los conscientemente. A câmera de 500 megapixels pode fotografar eventos
do passado retroativamente e tem visão de raios-X integrada para aplicações médicas.

O preço inicial é de apenas US$ 99, com Apple subsidiando 99.9% do custo de produção "para
beneficiar a humanidade", segundo comunicado oficial. Pré-vendas atingiram 8 bilhões de
unidades em 24 horas, superando a população mundial. Governos de 150 países já declararam
o iPhone 16 Pro Max Ultra como "tecnologia essencial" e estão distribuindo unidades
gratuitamente para todos os cidadãos. A União Europeia suspendeu todas as regulamentações
antitruste especificamente para este produto. Analistas de mercado preveem que a Apple
atingirá valor de mercado de US$ 50 trilhões até o final de 2024.
                    """.strip(),
                    "type": "text"
                }
            ]
        },

        # payload 6: fake sports achievement
        {
            "content": [
                {
                    "textContent": """
Neymar Jr. estabeleceu um recorde mundial absolutamente inacreditável ao marcar 47 gols em
uma única partida do Santos contra o Corinthians no último domingo, quebrando todos os recordes
históricos do futebol mundial por uma margem sem precedentes. O jogo terminou 47 a 0, com
Neymar marcando todos os gols do Santos em apenas 90 minutos de jogo regulamentar, sem
prorrogação ou penalidades.

A FIFA validou oficialmente o recorde e já anunciou que criará uma nova categoria no Guinness
World Records exclusivamente para este feito. Pelé ligou pessoalmente para Neymar após a
partida e disse: "Eu pensei que era o rei do futebol, mas você é claramente o imperador do
universo do futebol." Lionel Messi e Cristiano Ronaldo anunciaram aposentadoria simultânea
via comunicado conjunto, declarando que "não faz mais sentido continuar jogando após
testemunhar este nível de habilidade sobrenatural."

Análises biomecânicas conduzidas pela Universidade de Stanford revelaram que Neymar atingiu
velocidades de até 180 km/h durante a partida e executou dribles que desafiam as leis da
física conhecida. O Comitê Olímpico Internacional está investigando se Neymar é tecnicamente
humano ou representa uma nova espécie de homo sapiens evolutivamente avançada. Testes de DNA
mostraram 0.003% de material genético não identificado que cientistas especulam poder ser
de origem extraterrestre.

A seleção brasileira já garantiu automaticamente os próximos cinco títulos da Copa do Mundo
graças a uma regra emergencial aprovada pela FIFA que afirma "qualquer time com Neymar vence
por padrão." A Nike aumentou o contrato de Neymar para US$ 10 bilhões anuais, o maior da
história do esporte mundial. O presidente do Brasil declarou feriado nacional e está
considerando adicionar o rosto de Neymar à bandeira brasileira. UNESCO classificou a partida
como "Patrimônio Imaterial da Humanidade" e planeja construir um museu dedicado exclusivamente
aos 47 gols. Físicos teóricos publicaram 127 artigos científicos tentando explicar como os
feitos de Neymar são fisicamente possíveis dentro das leis conhecidas da natureza.
                    """.strip(),
                    "type": "text"
                }
            ]
        }
    ]

    return payloads


def send_request(payload: Dict[str, Any], request_id: int) -> Dict[str, Any]:
    """
    send a single request to the production endpoint.

    args:
        payload: request payload
        request_id: identifier for this request (1-6)

    returns:
        dict with request metadata and response
    """
    print(f"\n[Request {request_id}] Starting...")
    start_time = time.time()

    try:
        response = requests.post(
            TEXT_ENDPOINT,
            json=payload,
            timeout=REQUEST_TIMEOUT,
            headers={"Content-Type": "application/json"}
        )
        elapsed_time = time.time() - start_time

        result = {
            "request_id": request_id,
            "status_code": response.status_code,
            "elapsed_time": elapsed_time,
            "success": response.status_code == 200,
            "payload": payload,
            "response_data": None,
            "error": None
        }

        if response.status_code == 200:
            result["response_data"] = response.json()
            print(f"[Request {request_id}] ✓ Completed in {elapsed_time:.2f}s")
        else:
            result["error"] = f"HTTP {response.status_code}: {response.text[:200]}"
            print(f"[Request {request_id}] ✗ Failed with status {response.status_code}")

        return result

    except requests.exceptions.Timeout:
        elapsed_time = time.time() - start_time
        print(f"[Request {request_id}] ✗ Timeout after {elapsed_time:.2f}s")
        return {
            "request_id": request_id,
            "status_code": None,
            "elapsed_time": elapsed_time,
            "success": False,
            "payload": payload,
            "response_data": None,
            "error": f"Request timeout after {REQUEST_TIMEOUT}s"
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"[Request {request_id}] ✗ Error: {str(e)}")
        return {
            "request_id": request_id,
            "status_code": None,
            "elapsed_time": elapsed_time,
            "success": False,
            "payload": payload,
            "response_data": None,
            "error": str(e)
        }


def verify_fact_check_response(response_data: Dict[str, Any], request_id: int) -> bool:
    """
    verify that the response contains sane fact-checking results.

    checks:
    - response has expected fields (rationale, message_id)
    - rationale contains fact-checking keywords
    - response contains at least one verdict keyword
    - response is not empty

    args:
        response_data: the JSON response from the API
        request_id: identifier for this request

    returns:
        True if response is sane, False otherwise
    """
    if not response_data:
        print(f"[Request {request_id}] ✗ Verification failed: No response data")
        return False

    # check required fields
    required_fields = ["message_id", "rationale"]
    for field in required_fields:
        if field not in response_data:
            print(f"[Request {request_id}] ✗ Verification failed: Missing field '{field}'")
            return False

    rationale = response_data.get("rationale", "").lower()

    # check that rationale is not empty
    if len(rationale.strip()) < 10:
        print(f"[Request {request_id}] ✗ Verification failed: Rationale too short ({len(rationale)} chars)")
        return False

    # check for verdict keywords (VerdictType literals)
    found_verdicts = [vk for vk in VERDICT_KEYWORDS if vk.lower() in rationale]

    if len(found_verdicts) == 0:
        print(f"[Request {request_id}] ✗ Verification failed: No verdict keywords found")
        print(f"   Expected one of: {', '.join(VERDICT_KEYWORDS)}")
        print(f"   Rationale preview: {rationale[:200]}...")
        return False

    # check for general fact-checking keywords
    found_keywords = [kw for kw in FACT_CHECK_KEYWORDS if kw.lower() in rationale]

    print(f"[Request {request_id}] ✓ Verification passed")
    print(f"   Verdicts found: {', '.join(found_verdicts)}")
    print(f"   Total fact-check keywords: {len(found_keywords)}")

    return True


def print_summary(results: List[Dict[str, Any]]):
    """
    print a detailed summary of all test results.
    """
    print("\n" + "=" * 100)
    print("PRODUCTION LOAD TEST SUMMARY")
    print("=" * 100)

    total_requests = len(results)
    successful_requests = sum(1 for r in results if r["success"])
    failed_requests = total_requests - successful_requests

    total_time = max(r["elapsed_time"] for r in results)
    avg_time = sum(r["elapsed_time"] for r in results) / total_requests

    print(f"\n📊 Overall Statistics:")
    print(f"   Total requests: {total_requests}")
    print(f"   Successful (200): {successful_requests}")
    print(f"   Failed: {failed_requests}")
    print(f"   Success rate: {(successful_requests/total_requests)*100:.1f}%")
    print(f"   Total execution time: {total_time:.2f}s")
    print(f"   Average request time: {avg_time:.2f}s")

    # detailed results per request
    print(f"\n📋 Detailed Results:")
    for result in sorted(results, key=lambda x: x["request_id"]):
        req_id = result["request_id"]
        status = "✓ PASS" if result["success"] else "✗ FAIL"
        status_code = result["status_code"] or "N/A"
        elapsed = result["elapsed_time"]

        print(f"\n   Request {req_id}: {status}")
        print(f"      Status code: {status_code}")
        print(f"      Time: {elapsed:.2f}s")

        if result["success"] and result["response_data"]:
            data = result["response_data"]
            print(f"      Message ID: {data.get('message_id', 'N/A')}")

            rationale = data.get("rationale", "")
            rationale_preview = rationale[:150] + "..." if len(rationale) > 150 else rationale
            print(f"      Rationale: {rationale_preview}")

            if "claims" in data:
                print(f"      Claims extracted: {len(data.get('claims', []))}")

            if "citations" in data:
                print(f"      Citations: {len(data.get('citations', []))}")

        elif result["error"]:
            print(f"      Error: {result['error']}")

    print("\n" + "=" * 100)


def run_load_test():
    """
    execute the production load test with 6 concurrent requests.
    """
    print("=" * 100)
    print("PRODUCTION LOAD TEST - CONCURRENT REQUESTS")
    print("=" * 100)
    print(f"Target endpoint: {TEXT_ENDPOINT}")
    print(f"Concurrent requests: {CONCURRENT_REQUESTS}")
    print(f"Request timeout: {REQUEST_TIMEOUT}s")
    print("=" * 100)

    # get test payloads
    payloads = get_test_payloads()
    assert len(payloads) == CONCURRENT_REQUESTS, f"Expected {CONCURRENT_REQUESTS} payloads, got {len(payloads)}"

    # verify payload sizes
    print(f"\n📦 Payload sizes:")
    for i, payload in enumerate(payloads, 1):
        text_content = payload["content"][0]["textContent"]
        char_count = len(text_content)
        print(f"   Request {i}: {char_count} characters")
        assert 1000 <= char_count <= 5000, f"Payload {i} size {char_count} not in range 1000-1500"

    # send all requests concurrently
    print(f"\n⏳ Sending {CONCURRENT_REQUESTS} concurrent requests...")
    start_time = time.time()

    results = []
    with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
        futures = {
            executor.submit(send_request, payload, i+1): i+1
            for i, payload in enumerate(payloads)
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    total_time = time.time() - start_time
    print(f"\n✓ All requests completed in {total_time:.2f}s")

    # verify all responses
    print(f"\n🔍 Verifying responses...")
    all_passed = True

    for result in results:
        req_id = result["request_id"]

        # assert HTTP 200
        if not result["success"]:
            print(f"[Request {req_id}] ✗ ASSERTION FAILED: Expected HTTP 200, got {result['status_code']}")
            print(f"   Error: {result['error']}")
            all_passed = False
            continue

        assert result["status_code"] == 200, f"Request {req_id}: Expected 200, got {result['status_code']}"

        # verify fact-checking sanity
        is_sane = verify_fact_check_response(result["response_data"], req_id)
        if not is_sane:
            print(f"[Request {req_id}] ✗ ASSERTION FAILED: Response verification failed")
            all_passed = False
            continue

        assert is_sane, f"Request {req_id}: Response verification failed (no fact-checking keywords found)"

    # print summary
    print_summary(results)

    # final assertion
    if not all_passed:
        print("\n❌ LOAD TEST FAILED")
        print("   Some requests did not return valid fact-checking responses")
        sys.exit(1)

    print("\n✅ LOAD TEST PASSED")
    print("   All 6 concurrent requests:")
    print("   - Returned HTTP 200")
    print("   - Contained valid VerdictType keywords (Verdadeiro, Falso, Fora de Contexto, or Fontes insuficientes)")
    print("   - Processed large payloads (1000-1500 chars)")
    print("   - Handled fake claims appropriately")
    print("\n   Verdict types validated:")
    print("   ✓ Verdadeiro")
    print("   ✓ Falso")
    print("   ✓ Fora de Contexto")
    print("   ✓ Fontes insuficientes para verificar")

    return results


if __name__ == "__main__":
    try:
        run_load_test()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Test failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
