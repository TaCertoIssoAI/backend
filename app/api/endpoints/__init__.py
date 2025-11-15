# Importar apenas scraping por enquanto para evitar dependências
from . import scraping
# from . import text  # Descomentar quando precisar do pipeline completo

__all__ = ["scraping"]  # "text" removido temporariamente

