# Importar apenas scraping por enquanto para evitar dependências
from . import scraping, test, text, research
# from . import text  # Descomentar quando precisar do pipeline completo

__all__ = ["scraping", "test", "text", "research"]

