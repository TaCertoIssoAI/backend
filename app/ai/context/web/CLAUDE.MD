# Google Custom Search JSON API

This file explains how the project uses the Google Custom Search JSON API to programmatically search the web using FastAPI in Python. Sensitive keys are stored in `.env` and should never be version-controlled.

## Environment Variables

```env
GOOGLE_SEARCH_API_KEY=your_google_search_api_key
GOOGLE_CSE_CX=your_cse_cx_key
````

* `GOOGLE_SEARCH_API_KEY`: your Google Custom Search API key.
* `GOOGLE_CSE_CX`: Programmable Search Engine ID.

In the code, we use these variables via `os.environ` or through the project settings system.

Simple example:

```python
import os

GOOGLE_SEARCH_API_KEY = os.environ["GOOGLE_SEARCH_API_KEY"]
GOOGLE_CSE_CX = os.environ["GOOGLE_CSE_CX"]
```

## Base Endpoint

All requests are made to:

```text
https://www.googleapis.com/customsearch/v1
```

Minimum parameters for a search:

* `key`: `GOOGLE_SEARCH_API_KEY`
* `cx`: `GOOGLE_CSE_CX`
* `q`: search query string

Conceptual URL example:

```text
https://www.googleapis.com/customsearch/v1
  ?key=GOOGLE_SEARCH_API_KEY
  &cx=GOOGLE_CSE_CX
  &q=vacina+causa+autismo
```

## Python Search Function

Example of a Python wrapper using `httpx`:

```python
import os
from typing import Any, Dict, List, Optional

import httpx

GOOGLE_SEARCH_API_KEY = os.environ["GOOGLE_SEARCH_API_KEY"]
GOOGLE_CSE_CX = os.environ["GOOGLE_CSE_CX"]
GOOGLE_SEARCH_BASE_URL = "https://www.googleapis.com/customsearch/v1"


class GoogleSearchError(Exception):
    pass


async def google_search(
    query: str,
    *,
    num: int = 10,
    start: int = 1,
    site_search: Optional[str] = None,
    site_search_filter: Optional[str] = None,  # "i" include, "e" exclude
    date_restrict: Optional[str] = None,       # e.g., "d7", "m1"
    sort: Optional[str] = None,                # e.g., "date:r:20240101:20241231"
    file_type: Optional[str] = None,           # e.g., "pdf"
    safe: Optional[str] = None,                # "active" or "off"
    language: Optional[str] = None,            # e.g., "lang_pt"
) -> List[Dict[str, Any]]:
    """
    performs a search using google custom search api and returns the list of result items
    """

    params: Dict[str, Any] = {
        "key": GOOGLE_SEARCH_API_KEY,
        "cx": GOOGLE_CSE_CX,
        "q": query,
        "num": num,
        "start": start,
    }

    if site_search:
        params["siteSearch"] = site_search
    if site_search_filter:
        params["siteSearchFilter"] = site_search_filter
    if date_restrict:
        params["dateRestrict"] = date_restrict
    if sort:
        params["sort"] = sort
    if file_type:
        params["fileType"] = file_type
    if safe:
        params["safe"] = safe
    if language:
        params["lr"] = language

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(GOOGLE_SEARCH_BASE_URL, params=params)

    if response.status_code != 200:
        raise GoogleSearchError(
            f"google search error: {response.status_code} {response.text}"
        )

    data = response.json()
    return data.get("items", [])
```

all comments start with lowercase letters, as agreed.

## Integration with FastAPI

Example of a FastAPI endpoint that exposes the search function:

```python
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/search", tags=["search"])


@router.get("/")
async def search_endpoint(
    q: str = Query(..., description="search text"),
    num: int = Query(5, ge=1, le=10),
    site: Optional[str] = Query(None, description="domain filter, e.g., who.int"),
    last_days: Optional[int] = Query(None, ge=1, description="limit to the last N days"),
) -> Dict[str, Any]:
    """
    search endpoint that wraps the google custom search api
    """

    date_restrict = None
    if last_days is not None:
        date_restrict = f"d{last_days}"

    try:
        items = await google_search(
            query=q,
            num=num,
            site_search=site,
            site_search_filter="i" if site else None,
            date_restrict=date_restrict,
        )
    except GoogleSearchError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    # here you can normalize the fields relevant to your project
    results: List[Dict[str, Any]] = []
    for item in items:
        results.append(
            {
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
                "displayLink": item.get("displayLink"),
            }
        )

    return {"query": q, "results": results}
```

## Supported Filters and Recommended Usage

Below is a summary of the most useful filters for the project.

### 1. Site Filter

Two main approaches:

#### 1.1 `siteSearch` and `siteSearchFilter` Parameters

* `siteSearch`: domain to filter by, e.g., `who.int`
* `siteSearchFilter`:

  * `"i"` to include only that domain
  * `"e"` to exclude that domain

Example:

```python
items = await google_search(
    query="vaccines cause autism",
    site_search="who.int",
    site_search_filter="i",
)
```

#### 1.2 `site:` Operator in Query String

Another option is to include `site:` directly in the query:

```python
items = await google_search(
    query="vaccines cause autism site:who.int",
)
```

For multiple domains:

```python
items = await google_search(
    query="vaccines cause autism (site:who.int OR site:cdc.gov)",
)
```

In general, to keep the code cleaner, prefer `siteSearch` and `siteSearchFilter` when the domain filter comes from config or structured input.

### 2. Date Filters

#### 2.1 `dateRestrict` (Relative Range)

Possible formats:

* `dN` for the last N days, e.g., `d7`
* `wN` for the last N weeks, e.g., `w4`
* `mN` for the last N months, e.g., `m1`
* `yN` for the last N years, e.g., `y1`

Example, last 7 days:

```python
items = await google_search(
    query="vaccines cause autism",
    date_restrict="d7",
)
```

#### 2.2 `sort=date:r:YYYYMMDD:YYYYMMDD` (Absolute Range)

To limit results to an explicit date range:

```python
items = await google_search(
    query="vaccines cause autism",
    sort="date:r:20240101:20241231",
)
```

Format:

* `sort = "date:r:START:END"`
* dates in `YYYYMMDD` format

You can generate these dates from `datetime` objects in Python.

### 3. Language

Use the `lr` parameter to restrict by language:

* `lr="lang_pt"` for Portuguese
* `lr="lang_en"` for English
* etc.

Example:

```python
items = await google_search(
    query="vaccines cause autism",
    language="lang_pt",
)
```

### 4. Safe Search

Use the `safe` parameter:

* `"active"` to enable safe search
* `"off"` to disable it

Example:

```python
items = await google_search(
    query="vaccines cause autism",
    safe="active",
)
```

### 5. File Type

Use `fileType` to filter by file type:

* example: `fileType="pdf"`

```python
items = await google_search(
    query="vaccines cause autism",
    file_type="pdf",
)
```

You can also reinforce this in the query string:

```python
items = await google_search(
    query="vaccines cause autism filetype:pdf",
)
```

### 6. Pagination

The API supports pagination via:

* `num`: number of results per call, max 10
* `start`: index of the first result, starting at 1

Example, get the second page of 10 results:

```python
items = await google_search(
    query="vaccines cause autism",
    num=10,
    start=11,
)
```

You can build an internal pagination layer in the project to translate pages 1, 2, 3 into `start` values.

## Project Best Practices

* never log the actual API key (`GOOGLE_SEARCH_API_KEY`)
* ensure the `.env` file is never committed to the repository
* if the key leaks, revoke and generate a new one in the Google Console
* centralize the `google_search` function in this module so that any change to parameters or filtering policy is handled in one place


Você pode adicionar esse exemplo de saída JSON bruta ao final do seu arquivo de documentação como uma nova seção. Aqui está a versão traduzida e integrada da sugestão, com título e contexto explicando o uso prático para parsing:

## Example of Raw API Response

Below is a simplified example of a real response returned by the Google Custom Search API, so that developers can understand the structure and fields for JSON parsing:

```json
{
  "kind": "customsearch#search",
  "searchInformation": {
    "searchTime": 0.45,
    "formattedSearchTime": "0.45",
    "totalResults": "225000",
    "formattedTotalResults": "225,000"
  },
  "items": [
    {
      "kind": "customsearch#result",
      "title": "Lauro Jardim: Inaugurada por Michelle Bolsonaro, Casa do Autista ...",
      "link": "https://www.facebook.com/jornaloglobo/posts/1177801651049989/",
      "displayLink": "www.facebook.com",
      "snippet": "Aug 18, 2025 ... preso, e o Bolsonaro seria o Presidente por legitimidade !...",
      "pagemap": {
        "metatags": [
          {
            "og:title": "O Globo",
            "og:description": "Casa do Autista em BC é contestada em auditoria"
          }
        ]
      }
    },
    {
      "kind": "customsearch#result",
      "title": "'MARCHA ESTRANHA' | O autismo é uma condição do ...",
      "link": "https://www.instagram.com/p/DMNGzWuMVTW/",
      "displayLink": "www.instagram.com",
      "snippet": "Jul 17, 2025 ... Bolsonaro preso | O presidente dos Estados Unidos, Donald Trump..."
    }
    // ...more items
  ]
}
````

### Tip

To extract the most relevant fields, you can access:

```python
for item in data["items"]:
    title = item["title"]
    link = item["link"]
    snippet = item.get("snippet", "")
    display_link = item.get("displayLink", "")
```

This structure should be sufficient for most use cases where the goal is to list and display search results on your frontend or log them for analysis.


