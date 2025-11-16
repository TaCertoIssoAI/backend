"""
quick script to debug encoding issues with web scraping.
run this to see if content is being extracted correctly.
"""

import asyncio
import sys

# add parent directory to path
sys.path.insert(0, '/Users/caue.lira/Desktop/facul/TaCertoIssoAI')

from app.ai.context.web.apify_utils import scrapeGenericUrl


async def test_scraping():
    """test scraping the CNN Brasil URL that's having encoding issues"""
    
    url = "https://www.cnnbrasil.com.br/nacional/em-belem-cupula-dos-povos-cobra-participacao-popular-nas-acoes-climaticas/"
    
    print("=" * 80)
    print("ENCODING DEBUG TEST")
    print("=" * 80)
    print(f"Testing URL: {url}")
    print()
    
    # scrape the URL
    result = await scrapeGenericUrl(url)
    
    print(f"\nSuccess: {result['success']}")
    print(f"Error: {result.get('error', 'None')}")
    
    content = result.get('content', '')
    print(f"\nContent length: {len(content)} chars")
    print(f"Content type: {type(content)}")
    
    # check first 100 chars for encoding issues
    sample = content[:100] if len(content) >= 100 else content
    
    print(f"\nFirst 100 chars (repr):")
    print(repr(sample))
    
    print(f"\nFirst 100 chars (printed):")
    print(sample)
    
    # count problematic characters
    replacement_chars = sample.count('�')
    non_printable = sum(1 for c in sample if not c.isprintable() and c not in '\n\r\t ')
    non_ascii = sum(1 for c in sample if ord(c) > 127)
    
    print(f"\nCharacter analysis (first 100 chars):")
    print(f"  Replacement chars (�): {replacement_chars}")
    print(f"  Non-printable: {non_printable}")
    print(f"  Non-ASCII: {non_ascii}")
    
    # check if content looks readable
    if replacement_chars > 5 or non_printable > 20:
        print("\n⚠️  WARNING: Content has encoding issues!")
        print("This suggests:")
        print("  - Content might be compressed but not decompressed")
        print("  - Wrong encoding being used")
        print("  - Binary data being treated as text")
    else:
        print("\n✅ Content looks OK")
        print(f"\nClean preview (first 300 chars):")
        clean = ' '.join(content.split())
        print(clean[:300] + "...")


if __name__ == "__main__":
    asyncio.run(test_scraping())

