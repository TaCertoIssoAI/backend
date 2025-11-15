# Fake News Detector - WhatsApp Bot

A WhatsApp chatbot for fact-checking and fake news detection using multimodal AI and fact-checking APIs.

## 🏆 Special Achievement
 
I am very happy and proud to share with you a new conquest:

**1st place at RAIA's Hackathon 2025**, whose theme was "AI for Information Quality: Understanding, Awareness and Trust in the Digital World".

In a world surrounded and driven by information, this subject shows a light on the right point: the fight against disinformation is an urgent challenge. Fake News are now even harder to detect and easier to share. Deep fakes, AI-generated images, texts and voices, automated bots and so on turn the work against misinformation through the internet into a real and deep problem nowadays. Remember: fake news can kill people (as seen during the COVID-19 pandemic), destroy nations and lead people to unhealthy and nonsense habits.

So, congratulations and special thanks to **RAIA - Rede de Avanço em Inteligência Artificial** for bringing up efforts to this cause, and to **Brasil Monks** for believing in it and turning this event possible.

Thinking about this case, our team led a project focused on the core of the phenomenon: **WhatsApp**. The place where the majority of fake news is shared, a fact which is powered by encryption and the difficulty of tracing its source. So, we created an automated bot for WhatsApp using an integration between LLM's, WhatsApp and Google Fact Check APIs, which can magnificently tell in an easy way to the user if information is true, false or overblown. Also, the bot can say why it is not real and show real sources about the information. It doesn't matter what kind of information it is: text? links? voices? images? The bot is based on multimodal AI and can analyze everything.

It was 12 hours of deep immersion in this application with my teammates passing through mountains of problem-solving and leading a real integration between every component of the team, each one in their mastered abilities, which result could be not other but the first place of the competition. Thank you so much.

## Table of Contents
- [Fake News Detector - WhatsApp Bot](#fake-news-detector---whatsapp-bot)
  - [🏆 Special Achievement](#-special-achievement)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Architecture](#architecture)
  - [Requirements](#requirements)
  - [How to Use (Super Easy!)](#how-to-use-super-easy)
    - [Automatic Installation](#automatic-installation)
    - [Manual Execution (Advanced)](#manual-execution-advanced)
  - [Project Structure](#project-structure)
  - [API Endpoints](#api-endpoints)
    - [Core Endpoints](#core-endpoints)
      - [`POST /api/text`](#post-apitext)
      - [`POST /api/images`](#post-apiimages)
      - [`POST /api/multimodal`](#post-apimultimodal)
    - [Test Endpoints](#test-endpoints)
      - [`GET /api/test-adjudicator`](#get-apitest-adjudicator)
      - [`GET /api/test-evidence-retrieval`](#get-apitest-evidence-retrieval)
    - [System Endpoints](#system-endpoints)
      - [`GET /`](#get-)
      - [`GET /health`](#get-health)
  - [Development](#development)
  - [Features](#features)

## Introduction

**Fake News Detector** is a comprehensive fact-checking system that:
- Extracts verifiable claims from Portuguese text
- Retrieves evidence from trusted sources
- Uses LLM adjudication to provide verdicts with citations
- Supports text and multimodal analysis

## Architecture

**4-Step Pipeline:**
1. **Claim Extraction** - OpenAI LLM extracts verifiable claims from user input
2. **Evidence Retrieval** - Google Fact-Check API and other sources gather evidence
3. **LLM Adjudication** - Structured analysis produces verdicts with citations
4. **Response Formatting** - Clean API responses for WhatsApp integration

## Requirements

- [**Python 3.11+**](https://www.python.org/downloads/)
- OpenAI API Key
- Google Fact-Check API Key (optional)

**OR** (Recommended for Production):

- [**Docker**](https://docs.docker.com/get-docker/) and Docker Compose
- OpenAI API Key

## How to Use

### 🐳 Docker (Recommended)

**Quick Start:**
```bash
# 1. Configure environment
cp env.example .env
# Edit .env and add your OPENAI_API_KEY

# 2. Start everything
./docker-start.sh

# 3. Test
./docker-test.sh
```

**Access:**
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Selenium Grid: http://localhost:4444

📚 **Full Docker documentation**: See [DOCKER.md](DOCKER.md) and [QUICKSTART.md](QUICKSTART.md)

### Manual Execution

If you want to run manually or customize parameters:

```bash
# 1. Clone the repository
git clone https://github.com/lfelipediniz/TaCertoIssoAI
cd TaCertoIssoAI

# 2. Set up the environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# 4. Run the application
guvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Required environment variables:**
```env
# AI Services
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```
## Project Structure

```
app/
├── main.py              # FastAPI application entry point
├── api/
│   └── endpoints/       # API route handlers
│       ├── text.py      # Text analysis endpoints
│       ├── images.py    # Image analysis endpoints
│       └── multimodal.py # Combined analysis endpoints
├── core/
│   └── config.py        # Configuration management
├── models/
│   ├── schemas.py       # API request/response models
│   └── factchecking.py  # Pipeline data models
└── ai/
    ├── claim_extractor.py    # Step 1: Extract claims from text
    ├── evidence_retrieval.py # Step 2: Evidence retrieval
    ├── adjudicator.py        # Step 3: LLM fact-checking
    └── pipeline.py           # Complete pipeline orchestration
```

## API Endpoints

### Core Endpoints

#### `POST /api/text`
Analyze text-only messages for fact-checking.

**Request:**
```json
{
  "text": "The government announced new policies yesterday"
}
```

**Response:**
```json
{
  "message_id": "fc_123456",
  "verdict": "true",
  "rationale": "Government announcement confirmed by official sources",
  "citations": [
    {
      "url": "https://example.com/source",
      "title": "Official Government Statement",
      "publisher": "Government Portal",
      "quoted": "New policies were announced on the specified date"
    }
  ],
  "processing_time_ms": 3500
}
```

#### `POST /api/images`
Analyze images using OCR for fact-checking.

**Request (multipart/form-data):**
- `file`: Image file
- `chatId`: Optional chat identifier

**Response:** Same as `/api/text`

#### `POST /api/multimodal`
Analyze messages with both text and images.

**Request (multipart/form-data):**
- `text`: Optional text content
- `file`: Optional image file
- `chatId`: Optional chat identifier

**Response:** Same as `/api/text`

### Test Endpoints

#### `GET /api/test-adjudicator`
Test the LLM adjudication step with sample data.

#### `GET /api/test-evidence-retrieval`
Test evidence retrieval using Google Fact-Check API.

### System Endpoints

#### `GET /`
API information and version.

#### `GET /health`
Health check endpoint.

## Development

**Run tests:**
```bash
# Test individual components
curl -X GET "http://localhost:8000/api/test-adjudicator"
curl -X GET "http://localhost:8000/api/test-evidence-retrieval"

# Test full pipeline
curl -X POST "http://localhost:8000/api/text" \
  -H "Content-Type: application/json" \
  -d '{"text": "A vacina X causa infertilidade em mulheres"}'
```

## Features

- **Portuguese Language Support** - Optimized for pt-BR content
- **Structured LLM Outputs** - Type-safe Pydantic models throughout
- **Async-First Design** - High-performance async operations
- **Evidence Citations** - Transparent source attribution
- **Multi-Step Pipeline** - Modular claim extraction and verification
- **WhatsApp Integration** - Evolution API webhook support

---

**Fight against Fake News! 🚀📰**
