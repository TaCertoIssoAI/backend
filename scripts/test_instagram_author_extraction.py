#!/usr/bin/env python3
"""
quick test to demonstrate author extraction from instagram search results.
uses the sample data provided by the user.
"""

import json
import re
from typing import Dict, Any


def extract_author_info(item: Dict[str, Any]) -> Dict[str, str]:
    """
    extract instagram author information from search result.

    args:
        item: search result item dictionary

    returns:
        dict with author information (username, display_name, user_id)
    """
    author_info = {
        'username': None,
        'display_name': None,
        'user_id': None,
    }

    # try to get from pagemap metatags
    if 'pagemap' in item and 'metatags' in item['pagemap']:
        metatags = item['pagemap']['metatags'][0] if item['pagemap']['metatags'] else {}

        # extract user_id
        author_info['user_id'] = metatags.get('instapp:owner_user_id')

        # extract username and display name from og:title
        # format: "Display Name on Instagram: ..." or "Display Name (@username) • Instagram photo"
        og_title = metatags.get('og:title', '')
        if og_title:
            # try pattern: "Display Name on Instagram:"
            match = re.match(r'^(.+?)\s+on Instagram:', og_title)
            if match:
                author_info['display_name'] = match.group(1).strip()

            # try pattern: "Display Name (@username)"
            match = re.search(r'\(@(\w+)\)', og_title)
            if match:
                author_info['username'] = match.group(1)

        # extract username from og:description if not found yet
        # format: "X likes, Y comments - username on Date: ..."
        if not author_info['username']:
            og_desc = metatags.get('og:description', '')
            match = re.search(r'- (\w+) on \w+ \d+, \d{4}:', og_desc)
            if match:
                author_info['username'] = match.group(1)

    return author_info


# sample data from user's JSON dump (first 3 results)
sample_items = [
    {
        "link": "https://www.instagram.com/p/B0E8_sQhObP/",
        "pagemap": {
            "metatags": [{
                "instapp:owner_user_id": "6266434",
                "og:title": "Marcos Mion on Instagram: \"Se hoje o autismo teve uma vitoria...",
                "og:description": "582K likes, 24K comments - marcosmion on July 18, 2019: \"Se hoje o autismo..."
            }]
        }
    },
    {
        "link": "https://www.instagram.com/p/DMvwHR9PTdu/",
        "pagemap": {
            "metatags": [{
                "instapp:owner_user_id": "9120850937",
                "og:title": "Autistas Brasil on Instagram: \"Lucelmo Lacerda finalmente saiu...",
                "og:description": "2,054 likes, 446 comments - autistasbrasil on July 30, 2025: \"Lucelmo Lacerda..."
            }]
        }
    },
    {
        "link": "https://www.instagram.com/p/DIjP4NSOBw3/",
        "pagemap": {
            "metatags": [{
                "instapp:owner_user_id": "11071040959",
                "og:title": "Partido Liberal | PL22 on Instagram: \"Em 2020, o presidente Jair Bolsonaro...",
                "twitter:title": "Partido Liberal | PL22 (@plnacional22) • Instagram photo",
                "og:description": "2,400 likes, 44 comments - plnacional22 on April 17, 2025: \"Em 2020..."
            }]
        }
    }
]


def main():
    """demonstrate author extraction."""
    print("=" * 80)
    print("INSTAGRAM AUTHOR EXTRACTION DEMONSTRATION")
    print("=" * 80)
    print()

    for idx, item in enumerate(sample_items, start=1):
        author = extract_author_info(item)

        print(f"POST #{idx}")
        print(f"  URL: {item['link']}")
        print(f"  Username: @{author['username'] or 'N/A'}")
        print(f"  Display Name: {author['display_name'] or 'N/A'}")
        print(f"  User ID: {author['user_id'] or 'N/A'}")
        print()

    print("=" * 80)
    print("\nKEY FIELDS USED FOR EXTRACTION:")
    print("  1. pagemap.metatags[0].instapp:owner_user_id  → User ID")
    print("  2. pagemap.metatags[0].og:title               → Display Name")
    print("  3. pagemap.metatags[0].og:description         → Username")
    print("  4. pagemap.metatags[0].twitter:title          → Alternative source")
    print("=" * 80)


if __name__ == "__main__":
    main()
