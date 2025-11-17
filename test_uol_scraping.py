"""
test script to verify if UOL URL is scraped with BeautifulSoup or Apify
"""
import asyncio
import logging
from app.ai.context.web.apify_utils import scrapeGenericUrl, detectPlatform

# configure logging to see detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_uol_url():
    """test if UOL URL uses BeautifulSoup simple scraping or falls back to Apify"""

    # URL to test
    url = "https://noticias.uol.com.br/colunas/jamil-chade/2025/11/17/conselho-da-onu-aprova-plano-de-trump-para-gaza-com-tropas-internacionais.htm"

    print("\n" + "="*80)
    print(f"TESTING URL: {url}")
    print("="*80 + "\n")

    # 1. check platform detection
    platform = detectPlatform(url)
    print(f"✓ Platform detected: {platform.value}")
    print(f"  → Expected: GENERIC (not social media)")
    print()

    # 2. run the scraping
    print("Starting scraping process...")
    print("  → Will try BeautifulSoup first (simple HTTP)")
    print("  → Will fallback to Apify if simple scraping fails")
    print()

    result = await scrapeGenericUrl(url)

    # 3. analyze results
    print("\n" + "="*80)
    print("SCRAPING RESULTS:")
    print("="*80 + "\n")

    print(f"Success: {result['success']}")
    print(f"Content length: {len(result['content'])} chars")
    print(f"Scraping method: {result.get('metadata', {}).get('scraping_method', 'UNKNOWN')}")
    print(f"Platform: {result.get('metadata', {}).get('platform', 'UNKNOWN')}")

    if result.get('error'):
        print(f"Error: {result['error']}")

    print()

    # 4. show first 500 chars of content
    if result['success'] and result['content']:
        print("="*80)
        print("CONTENT PREVIEW (first 500 chars):")
        print("="*80)
        print(result['content'][:500])
        print("...")
        print()

    # 5. final verdict
    print("="*80)
    print("VERDICT:")
    print("="*80)

    if result.get('metadata', {}).get('scraping_method') == 'simple_http':
        print("✓ SUCCESS: URL was scraped using BeautifulSoup (simple HTTP)")
        print("  → No Apify credits used")
        print("  → Fast and cost-effective")
    elif result.get('metadata', {}).get('platform') == 'generic_web':
        print("! FALLBACK: URL was scraped using Apify actor")
        print("  → Simple HTTP scraping failed")
        print("  → Used browser-based scraping (slower, uses Apify credits)")
    else:
        print("? UNKNOWN: Could not determine scraping method")

    print()

if __name__ == "__main__":
    asyncio.run(test_uol_url())
