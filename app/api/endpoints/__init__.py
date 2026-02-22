# Importar apenas scraping por enquanto para evitar dependências
from . import scraping, text
# from . import text  # Descomentar quando precisar do pipeline completo

__all__ = ["scraping", "text"]
