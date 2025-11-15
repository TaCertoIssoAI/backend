# Dockerfile para Backend com Selenium/Chrome integrado
# Container único para deploy no Render Free Tier

FROM python:3.11-slim

# Definir diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema + Chrome + ChromeDriver
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Dependências Python build
    gcc \
    g++ \
    libxml2-dev \
    libxslt-dev \
    libssl-dev \
    libffi-dev \
    # Dependências runtime
    curl \
    wget \
    gnupg \
    unzip \
    # Dependências Chrome
    libnss3 \
    libfontconfig1 \
    libxss1 \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libnspr4 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    xdg-utils \
    fonts-liberation \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Instalar Google Chrome
RUN wget -q -O /tmp/google-chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
    && apt-get update \
    && apt-get install -y /tmp/google-chrome.deb \
    && rm /tmp/google-chrome.deb \
    && rm -rf /var/lib/apt/lists/*

# Instalar ChromeDriver (versão estável conhecida)
RUN wget -q -O /tmp/chromedriver-linux64.zip https://storage.googleapis.com/chrome-for-testing-public/142.0.7444.162/linux64/chromedriver-linux64.zip \
    && unzip /tmp/chromedriver-linux64.zip -d /tmp/ \
    && mv /tmp/chromedriver-linux64/chromedriver /usr/local/bin/chromedriver \
    && rm -rf /tmp/chromedriver* \
    && chmod +x /usr/local/bin/chromedriver \
    && chromedriver --version

# Copiar requirements e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY . .

# Criar usuário não-root (mas Chrome precisa rodar como usuário normal)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Mudar para usuário não-root
USER appuser

# Expor porta
EXPOSE 8000

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000
# Configurar para usar Selenium LOCAL (não remoto)
ENV USE_SELENIUM_REMOTE=false
ENV SELENIUM_REMOTE_URL=http://localhost:4444/wd/hub

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando para iniciar a aplicação
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
