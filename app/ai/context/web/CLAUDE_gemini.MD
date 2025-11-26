## Introduction

This document explains how to integrate **Google Search Grounding** using **Gemini API models** to improve factual accuracy and generate citations based on real-time web content.

## Benefits

Using Google Search grounding allows you to:

- **Access up-to-date information**
- **Reduce hallucinations in model responses**
- **Include citations from reliable sources**
- **Integrate with various models in the Gemini family**

## How it Works

1. **Enable the google_search tool** in your API configuration.
2. **The model automatically determines** if a web search is needed based on the prompt.
3. **It generates and runs search queries** on the web.
4. **Processes results** and synthesizes a response.
5. **Returns a grounded response** with a `groundingMetadata` field.

## Example Usage (Python)

```python
from google import genai
from google.genai import types

client = genai.Client()

grounding_tool = types.Tool(
    google_search=types.GoogleSearch()
)

config = types.GenerateContentConfig(
    tools=[grounding_tool]
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Who won the euro 2024?",
    config=config,
)

print(response.text)
```

## Displaying Inline Citations

Use the following function to insert clickable inline links into the generated text:

```python
def add_citations(response):
    text = response.text
    supports = response.candidates[0].grounding_metadata.grounding_supports
    chunks = response.candidates[0].grounding_metadata.grounding_chunks

    sorted_supports = sorted(supports, key=lambda s: s.segment.end_index, reverse=True)

    for support in sorted_supports:
        end_index = support.segment.end_index
        if support.grounding_chunk_indices:
            citation_links = []
            for i in support.grounding_chunk_indices:
                if i < len(chunks):
                    uri = chunks[i].web.uri
                    citation_links.append(f"[{i + 1}]({uri})")
            citation_string = ", ".join(citation_links)
            text = text[:end_index] + citation_string + text[end_index:]

    return text
```

## Example Output with Citations

```
Spain won Euro 2024, defeating England 2-1 in the final.[1](https://...), [2](https://...)
```

## `groundingMetadata` Structure

```json
{
  "groundingMetadata": {
    "webSearchQueries": ["UEFA Euro 2024 winner"],
    "searchEntryPoint": {
      "renderedContent": "<!-- HTML and CSS -->"
    },
    "groundingChunks": [
      {"web": {"uri": "https://...", "title": "uefa.com"}},
      {"web": {"uri": "https://...", "title": "aljazeera.com"}}
    ],
    "groundingSupports": [
      {
        "segment": {"startIndex": 0, "endIndex": 85, "text": "..."},
        "groundingChunkIndices": [0]
      }
    ]
  }
}
```

## Compatible Models

| Model                | Supported |
|----------------------|-----------|
| Gemini 2.5 Pro       | ✔️        |
| Gemini 2.5 Flash     | ✔️        |
| Gemini 1.5 Pro       | ✔️        |
| Gemini 1.5 Flash     | ✔️        |

> Legacy models like Gemini 1.5 support `google_search_retrieval` with dynamic configuration.

## Pricing

Each **search query triggered by the model** is billed separately. If multiple queries are generated in a single API call, each one is charged.

Check current rates on the [Gemini API pricing page](https://cloud.google.com/vertex-ai/generative-ai/pricing).

## Final Notes

This feature is especially effective for improving response reliability in real-time applications, such as **Tá Certo Isso AI**, where trustworthy and up-to-date information is crucial.