from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import scraping, research
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title="Fake News Detector API - Web Scraping",
    description="API de Web Scraping com Apify integrado",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# rotas de scraping e research
app.include_router(scraping.router, prefix="/api", tags=["scraping"])
app.include_router(research.router, prefix="/api", tags=["research"])

# Descomentar quando precisar do pipeline completo:
# app.include_router(text.router, prefix="/api", tags=["text"])

@app.get("/")
async def root():
    return {"message": "Fake News Detector API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}