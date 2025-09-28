# Dockerfile para Sistema OCR Ultra-Rápido API
FROM python:3.12-slim

# Metadatos
LABEL maintainer="armandix23"
LABEL description="Sistema OCR Ultra-Rápido API con FastAPI"
LABEL version="2.0.0"

# Variables de entorno
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos de dependencias
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY . .

# Crear directorios necesarios
RUN mkdir -p entrada procesados resultados logs

# Exponer puerto
EXPOSE 8000

# Comando de inicio
CMD ["uvicorn", "api_ocr:app", "--host", "0.0.0.0", "--port", "8000"]
