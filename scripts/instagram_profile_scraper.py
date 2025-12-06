#!/usr/bin/env python3
"""
instagram profile scraper.

attempts to fetch instagram profile data using both the ?__a=1 parameter
(which may return JSON) and HTML parsing fallback.

note: instagram actively blocks automated requests. this script uses
proper headers but may still be blocked. for production use, consider
using official instagram api or third-party services.
"""

import os
import sys
import json
import re
import asyncio
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup


class InstagramScraper:
    """scraper for instagram profile information."""

    def __init__(self, timeout: float = 30.0):
        """
        initialize instagram scraper.

        args:
            timeout: request timeout in seconds
        """
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,pt-BR;q=0.8,pt;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Cache-Control": "max-age=0",
        }

    async def fetch_profile_json(self, username: str) -> Optional[Dict[str, Any]]:
        """
        attempt to fetch profile data using ?__a=1 parameter.

        this used to return JSON data but instagram may have disabled it.

        args:
            username: instagram username

        returns:
            dict with profile data if successful, None otherwise
        """
        url = f"https://www.instagram.com/{username}/?__a=1"

        print(f"[FETCH] trying JSON endpoint: {url}")

        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                response = await client.get(url, headers=self.headers)

            print(f"[FETCH] status code: {response.status_code}")
            print(f"[FETCH] content-type: {response.headers.get('content-type', 'unknown')}")

            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')

                # check if response is JSON
                if 'application/json' in content_type:
                    data = response.json()
                    print(f"[FETCH] received JSON response")
                    return data

                # instagram might return HTML even with ?__a=1
                print(f"[FETCH] received non-JSON response (probably HTML)")
                return None

            print(f"[FETCH] request failed with status {response.status_code}")
            return None

        except httpx.TimeoutException:
            print(f"[FETCH] request timed out after {self.timeout}s")
            return None
        except json.JSONDecodeError as e:
            print(f"[FETCH] failed to parse JSON: {e}")
            return None
        except Exception as e:
            print(f"[FETCH] error: {type(e).__name__}: {str(e)}")
            return None

    async def fetch_profile_html(self, username: str) -> Optional[str]:
        """
        fetch profile HTML page.

        args:
            username: instagram username

        returns:
            HTML content as string if successful, None otherwise
        """
        url = f"https://www.instagram.com/{username}/"

        print(f"[FETCH] fetching HTML from: {url}")

        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                response = await client.get(url, headers=self.headers)

            print(f"[FETCH] status code: {response.status_code}")
            print(f"[FETCH] content length: {len(response.text)} characters")

            if response.status_code == 200:
                return response.text

            print(f"[FETCH] request failed with status {response.status_code}")
            return None

        except httpx.TimeoutException:
            print(f"[FETCH] request timed out after {self.timeout}s")
            return None
        except Exception as e:
            print(f"[FETCH] error: {type(e).__name__}: {str(e)}")
            return None

    def parse_html_for_json(self, html: str) -> Optional[Dict[str, Any]]:
        """
        parse HTML to extract embedded JSON data.

        instagram embeds profile data in <script> tags as JSON.

        args:
            html: HTML content

        returns:
            extracted profile data dict if found, None otherwise
        """
        print(f"[PARSE] searching for embedded JSON in HTML")

        soup = BeautifulSoup(html, 'html.parser')

        # instagram embeds data in script tags with type="application/ld+json"
        # or in window._sharedData
        scripts = soup.find_all('script', type='application/ld+json')

        for script in scripts:
            try:
                data = json.loads(script.string)
                print(f"[PARSE] found JSON-LD data")
                return data
            except json.JSONDecodeError:
                continue

        # try to find window._sharedData
        all_scripts = soup.find_all('script')
        for script in all_scripts:
            if script.string and 'window._sharedData' in script.string:
                print(f"[PARSE] found window._sharedData")

                # extract the JSON part
                match = re.search(r'window\._sharedData\s*=\s*({.*?});', script.string)
                if match:
                    try:
                        data = json.loads(match.group(1))
                        return data
                    except json.JSONDecodeError as e:
                        print(f"[PARSE] failed to parse _sharedData: {e}")

        # try to find other embedded JSON
        for script in all_scripts:
            if script.string and script.string.strip().startswith('{'):
                try:
                    data = json.loads(script.string)
                    if 'graphql' in data or 'user' in data:
                        print(f"[PARSE] found embedded profile data")
                        return data
                except json.JSONDecodeError:
                    continue

        print(f"[PARSE] no embedded JSON data found")
        return None

    def extract_profile_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        extract useful profile information from instagram data.

        handles different data structures from JSON endpoint or HTML parsing.

        args:
            data: raw data dict from instagram

        returns:
            dict with extracted profile information
        """
        profile = {
            'username': None,
            'full_name': None,
            'biography': None,
            'followers': None,
            'following': None,
            'posts_count': None,
            'is_verified': None,
            'is_private': None,
            'profile_pic_url': None,
            'external_url': None,
        }

        # try different data structures

        # structure 1: graphql.user
        if 'graphql' in data and 'user' in data['graphql']:
            user = data['graphql']['user']
            profile.update({
                'username': user.get('username'),
                'full_name': user.get('full_name'),
                'biography': user.get('biography'),
                'followers': user.get('edge_followed_by', {}).get('count'),
                'following': user.get('edge_follow', {}).get('count'),
                'posts_count': user.get('edge_owner_to_timeline_media', {}).get('count'),
                'is_verified': user.get('is_verified'),
                'is_private': user.get('is_private'),
                'profile_pic_url': user.get('profile_pic_url_hd') or user.get('profile_pic_url'),
                'external_url': user.get('external_url'),
            })

        # structure 2: user (from ?__a=1 endpoint)
        elif 'user' in data:
            user = data['user']
            profile.update({
                'username': user.get('username'),
                'full_name': user.get('full_name'),
                'biography': user.get('biography'),
                'followers': user.get('follower_count') or user.get('edge_followed_by', {}).get('count'),
                'following': user.get('following_count') or user.get('edge_follow', {}).get('count'),
                'posts_count': user.get('media_count') or user.get('edge_owner_to_timeline_media', {}).get('count'),
                'is_verified': user.get('is_verified'),
                'is_private': user.get('is_private'),
                'profile_pic_url': user.get('profile_pic_url_hd') or user.get('profile_pic_url'),
                'external_url': user.get('external_url'),
            })

        # structure 3: entry_data.ProfilePage[0].graphql.user
        elif 'entry_data' in data and 'ProfilePage' in data['entry_data']:
            pages = data['entry_data']['ProfilePage']
            if pages and 'graphql' in pages[0] and 'user' in pages[0]['graphql']:
                user = pages[0]['graphql']['user']
                profile.update({
                    'username': user.get('username'),
                    'full_name': user.get('full_name'),
                    'biography': user.get('biography'),
                    'followers': user.get('edge_followed_by', {}).get('count'),
                    'following': user.get('edge_follow', {}).get('count'),
                    'posts_count': user.get('edge_owner_to_timeline_media', {}).get('count'),
                    'is_verified': user.get('is_verified'),
                    'is_private': user.get('is_private'),
                    'profile_pic_url': user.get('profile_pic_url_hd') or user.get('profile_pic_url'),
                    'external_url': user.get('external_url'),
                })

        return profile

    async def scrape_profile(self, username: str) -> Optional[Dict[str, Any]]:
        """
        scrape instagram profile using multiple methods.

        tries JSON endpoint first, then falls back to HTML parsing.

        args:
            username: instagram username (without @)

        returns:
            dict with profile information if successful, None otherwise
        """
        print(f"\n{'=' * 80}")
        print(f"SCRAPING INSTAGRAM PROFILE: @{username}")
        print(f"{'=' * 80}\n")

        # method 1: try JSON endpoint
        data = await self.fetch_profile_json(username)
        if data:
            profile = self.extract_profile_info(data)
            if any(profile.values()):
                print(f"[SUCCESS] extracted profile data from JSON endpoint")
                return profile

        # method 2: fetch and parse HTML
        html = await self.fetch_profile_html(username)
        if html:
            # save raw HTML for debugging
            html_file = f"{username}_profile.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"[INFO] saved raw HTML to: {html_file}")

            # try to extract JSON from HTML
            data = self.parse_html_for_json(html)
            if data:
                # save extracted JSON for debugging
                json_file = f"{username}_data.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"[INFO] saved extracted JSON to: {json_file}")

                profile = self.extract_profile_info(data)
                if any(profile.values()):
                    print(f"[SUCCESS] extracted profile data from HTML")
                    return profile

        print(f"[FAILURE] could not extract profile data")
        return None


async def main():
    """main execution function."""
    # get username from command line or use default
    if len(sys.argv) > 1:
        username = sys.argv[1].strip().lstrip('@')
    else:
        username = "messss.bar"
        print(f"[INFO] no username provided, using default: @{username}")
        print(f"[INFO] usage: python {sys.argv[0]} <username>\n")

    scraper = InstagramScraper(timeout=30.0)

    try:
        profile = await scraper.scrape_profile(username)

        if profile:
            print(f"\n{'=' * 80}")
            print(f"PROFILE INFORMATION")
            print(f"{'=' * 80}\n")

            for key, value in profile.items():
                if value is not None:
                    print(f"{key:20s}: {value}")

            # save profile to JSON
            output_file = f"{username}_profile.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(profile, f, indent=2, ensure_ascii=False)
            print(f"\n[INFO] profile saved to: {output_file}\n")

        else:
            print(f"\n[ERROR] failed to scrape profile for @{username}")
            print(f"[ERROR] instagram may be blocking automated requests")
            print(f"[ERROR] try opening the profile in a browser to verify it exists\n")
            sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {str(e)}\n")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
