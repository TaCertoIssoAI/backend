# CLAUDE.md

# Fake-News Detector - Project Overview (WhatsApp Chatbot)

## 1) Summary

A WhatsApp chatbot that receives user messages via Evolution API, extracts the central claim from text or media, runs a retrieval plus LLM adjudication pipeline, and replies with a verdict and citations. The backend is our codebase. The bot supports text-only messages, images with or without captions, and links.
 
---

## 2) Purpose & Motivation

* Problem: Misleading and false claims spread quickly across chats and groups. Manual verification is slow and requires expertise.
* Our aim: Deliver source-grounded fact checks directly inside WhatsApp with minimal friction.
* Principles:
  * Evidence over opinions with transparent citations.
  * Privacy by design with explicit user consent.
  * Clear, brief, and actionable replies.

---

## 3) What We Want to Achieve (Goals)

* G1 - One message to verify: users forward or paste content, the bot replies with a concise verdict.
* G2 - Hybrid context: combine message text, attached image OCR, and optional URL context for robust claim extraction.
* G3 - Grounded verdicts: return True or False or Misleading or Unverifiable with 3 to 5 citations and a short rationale.
* G4 - Trust and transparency: show what was analyzed, how we searched, and why we decided.
* G5 - Low latency: target P50 under 5 seconds for text, under 12 seconds when OCR is needed.
* G6 - Learn and improve: thumbs up or down feedback loop, disagreement handling, and continuous evaluation.

---

## 4) Core User Scenarios (Happy Paths)

1) Quick Check - text first  
User sends a sentence or claim. Bot replies with verdict, rationale, and citations.

2) Image Post - image first  
User sends a screenshot or photo with a claim. Backend runs OCR and merges caption plus OCR text. Bot replies with verdict and citations.

3) Link Check - URL first  
User sends a link. Backend fetches metadata, extracts the central statement, and replies with verdict and citations.

---

## 5) Non-Goals (V1)

* Deepfake detection for video or audio at production quality.
* Group moderation or automated takedowns.
* Cross-platform integrations outside WhatsApp.

---

## 6) Success Metrics (KPIs)

* Adoption: daily and weekly active users, requests per user, retention.
* Quality: helpfulness rating at least 4 out of 5, disagreement rate under 10 percent.
* Latency: P50 decision time for text and for OCR paths.
* Coverage: percentage of requests that return at least 3 independent sources.

---

## 7) Privacy, Consent, and Compliance

* Consent in chat: first-time onboarding message explains processing and asks for OK before any analysis.
* Redaction guidance: instruct users to remove personal data before sending. We do not require screenshots if text is enough.
* Data minimization: store only what is required for quality and audit. Prefer short text over full images when feasible.
* Retention policy: default 7 to 30 days with user-triggered deletion at any time.
* Controls: "delete my data", "stop analyzing my messages", domain blocking for URL fetch.
* Documentation: clear Privacy Policy and Data Use docs aligned with WhatsApp Business and Evolution API terms.

---

## 8) Tech Stack (Proposed)

### WhatsApp Connector

* Evolution API for webhooks and sending replies.
* Verified webhook endpoint with signature validation and retry handling.

### Backend

* **Framework**: FastAPI with Python 3.11+, deployable to Heroku/Render
* **API Layer**: Three main endpoints for content analysis:
  * `/api/text` - Text-only fact-checking
  * `/api/images` - Image analysis with OCR
  * `/api/multimodal` - Combined text and image processing
* **Processing**:
  * Claim extraction: heuristics and NLP to isolate the central statement
  * Retrieval: web and news search with quality filters, deduplication, and date normalization
  * Adjudication: LLM reasoning grounded strictly in retrieved evidence
* **Datastores**:
  * SQLAlchemy for database ORM (PostgreSQL/SQLite)
  * Redis for caching repeated claims and recent evidence
  * Optional object storage for media files
* **Configuration**: Environment-based settings without Pydantic dependencies
* **Validation**: Python dataclasses for request/response schemas
* **Observability**: Basic logging with configurable levels

### ML and AI

* OCR: on demand for images.
* NER and semantic parsing: entity and date normalization, language detection.
* LLM: provider-agnostic adapter that enforces citation grounding and structured outputs.

---

## 9) High-Level Architecture (Overview)

User on WhatsApp  
→ Evolution API (webhook event)  
→ Backend Ingest API  
→ Preprocess and Claim Builder (merge text, OCR when needed, normalize entities and dates)  
→ Retrieval Service (search, filter, dedupe, timestamp checks)  
→ LLM Adjudicator (verdict and citations)  
→ Response Builder (JSON payload)  
→ Evolution API send message  
→ User receives verdict with citations

Supporting services: admin auth, metrics and alerts, storage and retention jobs.

---

## 10) WhatsApp Permissions and Capabilities (V1)

* Receive message events from Evolution API webhooks.
* Send text replies that include links to sources.
* Receive media messages and fetch media with short-lived links from Evolution API.
* No group-wide scraping or background reading of chats. Only user-initiated messages.

---

## 11) Data We Receive and Process (Typical Payload)

* Required: message id, sender id, timestamp, message type.
* Optional structured: text body, caption, URL, quoted message text, language.
* Optional media: image bytes or media URL token for OCR, optional thumbnail.
* Client hints: bot version, feature flags.
* Privacy flags: user consent status, data retention preference.

---

## 12) API Endpoints - FastAPI Backend

### POST /api/text
Analyze text-only messages for fact-checking.

Body (JSON):
```json
{
  "text": "string (required, max 10000 chars)",
  "chatId": "string (optional)"
}
````

Response:

```json
{
  "message_id": "string",
  "verdict": "true|false|misleading|unverifiable",
  "confidence": 0.85,
  "rationale": "Brief explanation",
  "citations": [
    {
      "title": "Article title",
      "source": "Source name",
      "url": "https://...",
      "snippet": "Relevant excerpt",
      "published_at": "2024-01-01T00:00:00Z"
    }
  ],
  "processing_time_ms": 4500
}
```

### POST /api/images

Analyze image-only messages using OCR for fact-checking.

Body (multipart/form-data):

* `file`: Image file (required)
* `chatId`: String (optional)

Response: Same as `/api/text`

### POST /api/multimodal

Analyze messages containing both text and images.

Body (multipart/form-data):

* `text`: String (optional, max 10000 chars)
* `file`: Image file (optional)
* `chatId`: String (optional)

Response: Same as `/api/text`

### GET /

Root endpoint returning API info.

### GET /health

Health check endpoint.

---

## 13) WhatsApp UX Overview

1. Onboarding
   Bot welcomes the user and asks for consent. Explains privacy and how to use the service.

2. Send content
   User forwards a message, an image, or a link. Optional command: "check", "help", "delete my data".

3. Processing
   Bot acknowledges receipt and explains that citations will follow.

4. Result
   Bot sends verdict badge, short rationale, and 3 to 5 citations. Also sends "How we decided" link or short explainer.

5. Follow-ups
   Buttons or quick replies: "check another", "disagree", "learn more", "delete my data".

---

## 14) Performance and Reliability

* Fast path: skip OCR when text is available.
* Caching: normalize claim and domain, reuse recent adjudications.
* Timeouts: hard limits for retrieval and LLM. If timeouts occur, reply with Unverifiable and a brief explanation.
* Resilience: circuit breaker for external search and graceful degradation to cached sources.

---

## 15) Security Considerations

* Transport: HTTPS only with HSTS.
* Webhook verification: signature or secret validation for Evolution API events.
* Input limits: maximum image size and page fetch limits, antivirus scanning for media when stored.
* Abuse prevention: rate limits per sender, anomaly detection for spam or scripted requests.
* Data segregation: separate buckets for temporary artifacts with lifecycle rules.
* Logging: do not log raw media or full messages. Store hashed or truncated references only.

---

## 16) Accessibility and Internationalization

* Language support: English and Portuguese at launch with automatic language detection.
* Readability: short messages, clear verdict labels, and numbered citations.
* Right to left support on the roadmap.

---

## 17) Testing Strategy

* Unit tests: claim parsing, retrieval ranking, citation formatting, and date checks.
* Integration tests: end to end for text-only and image OCR flows.
* Real site checks: news outlets, social posts, and blogs.
* Compliance tests: consent, data deletion, retention timers, and opt-out flow.
* Load tests: spikes during breaking news events.

---

## 18) Analytics and Telemetry - Privacy Aware

* Event counts: requests received and completed, OCR rate, link rate.
* Quality signals: helpfulness rating, disagreement rate, coverage of citations.
* Error budgets: OCR failures, retrieval misses, LLM timeouts.
* Opt in only: no per-user tracking by default. Use sampled and hashed metrics.

---

## 19) Roadmap - Milestones

* M1 - Prototype: manual retrieval plus lightweight adjudication, English only, local dev sandbox, simple consent.
* M2 - Media support: OCR for images, better claim extraction, basic citations with date normalization.
* M3 - Quality pass: improved retrieval filters, caching, PT-BR enabled, better rationale formatting.
* M4 - Privacy and compliance: full consent flows, deletion requests, retention policies, documentation.
* M5 - Feedback loop: disagreement capture, reviewer tools, continuous offline eval.
* M6 - Scale and polish: performance tuning, reliability work, more locales.

---

## 20) Acceptance Criteria - V1

* First-time consent message is shown and stored before analysis.
* Text-only path average P50 latency 5 seconds or less.
* Image OCR path average P50 latency 12 seconds or less.
* Verdict message includes badge, short rationale, and at least 3 citations.
* Works for direct messages and small groups where the bot is added.
* Clear commands for help and data deletion.

---

## 21) Operations and Support

* Runbooks: retrieval provider down, LLM quota exceeded, webhook retries, media fetch failures.
* Monitoring: uptime, error rates, queue depth, latency SLOs.
* Updates: semantic versioning, staged rollout, changelog in the repo.
* Support: in-chat "Report an issue" that links to a form or GitHub issue template.

---

## 22) Open Questions

* Trusted sources list vs open web search for early versions.
* Default retention window for media and whether to store OCR text only after extraction.
* Offline or degraded mode behavior when search is rate limited.
* Group dynamics: how to handle multiple senders and avoid noisy replies.

---

### Appendix - Glossary

* Claim extraction: turning noisy chat text and OCR content into a single central statement to verify.
* Grounded reasoning: a verdict and rationale tied to citations users can open and read.
* Unverifiable: insufficient or conflicting evidence within time and search constraints.

---

### Code Style Guidelines

To ensure consistency across the codebase:

* **Comments**: must be short and objective. Avoid overcommenting or restating the obvious.
* **Language**: all code must be written in English.
* **Naming**: variables, functions and files must start with **lowercase** letters and use clear, descriptive names.
* **Comment style**: comments should also start with **lowercase** and only be used where they clarify real intent or edge cases.
* **No auto-generated docs**: do not generate `.md` files to explain code changes unless explicitly requested.

```

