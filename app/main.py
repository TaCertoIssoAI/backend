from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# APENAS SCRAPING - Comentar text para evitar erros de dependências
from app.api.endpoints import scraping
# from app.api.endpoints import text  # Descomente quando precisar do pipeline completo
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

# Apenas rota de scraping por enquanto
app.include_router(scraping.router, prefix="/api", tags=["scraping"])

# Descomentar quando precisar do pipeline completo:
# app.include_router(text.router, prefix="/api", tags=["text"])

@app.get("/")
async def root():
    return {"message": "Fake News Detector API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}