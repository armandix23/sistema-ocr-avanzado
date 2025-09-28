# Sistema OCR Ultra-Rápido y Robusto

Sistema de procesamiento OCR avanzado con clasificación automática, extracción semántica y validación inteligente para documentos en español.

## 🚀 Características

- **Modo Ultra-Rápido**: Procesamiento optimizado < 3 segundos/documento
- **Clasificación Automática**: Detecta automáticamente el tipo de documento
- **Extracción Semántica**: Sin dependencia de regex frágiles
- **Validación Inteligente**: Sistema de confianza y validación automática
- **Salida Unificada**: Estructura JSON estandarizada para todos los tipos
- **100% Automático**: Sin intervención humana, procesamiento por lotes

## 📋 Tipos de Documentos Soportados

- **FACTURA**: Facturas comerciales, recibos de compra
- **RECIBO**: Recibos bancarios, transferencias, pagos
- **RECIBO_AEROLINEA**: Tickets de aerolíneas, reservas de vuelos
- **MULTA**: Multas de tráfico, sanciones administrativas
- **CONTRATO**: Contratos, acuerdos, convenios
- **OTROS**: Documentos diversos

## 🛠️ Instalación

### Requisitos del Sistema
- **Python**: 3.12 o superior
- **RAM**: 4GB mínimo (8GB recomendado)
- **Espacio en disco**: 2GB para el sistema + espacio para documentos
- **Sistema operativo**: Windows 10/11, macOS 10.15+, Linux Ubuntu 20.04+

### Instalación de Python

#### Windows
1. **Descargar Python**:
   - Ve a [python.org/downloads](https://www.python.org/downloads/)
   - Descarga Python 3.12.x para Windows
   - **IMPORTANTE**: Marca "Add Python to PATH" durante la instalación

2. **Verificar instalación**:
   ```cmd
   python --version
   pip --version
   ```

#### macOS
1. **Usando Homebrew (recomendado)**:
   ```bash
   # Instalar Homebrew si no lo tienes
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   
   # Instalar Python
   brew install python@3.12
   ```

2. **Usando el instalador oficial**:
   - Ve a [python.org/downloads](https://www.python.org/downloads/)
   - Descarga Python 3.12.x para macOS
   - Ejecuta el instalador .pkg

3. **Verificar instalación**:
   ```bash
   python3 --version
   pip3 --version
   ```

#### Linux (Ubuntu/Debian)
```bash
# Actualizar sistema
sudo apt update

# Instalar Python 3.12
sudo apt install python3.12 python3.12-pip python3.12-venv

# Verificar instalación
python3.12 --version
pip3.12 --version
```

### Instalación del Proyecto

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/armandix23/sistema-ocr-avanzado.git
   cd sistema-ocr-avanzado
   ```

2. **Crear entorno virtual (recomendado)**:
   
   **Windows**:
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```
   
   **macOS/Linux**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instalar dependencias**:
   ```bash
   # Actualizar pip
   pip install --upgrade pip
   
   # Instalar dependencias principales
   pip install paddlepaddle
   pip install opencv-python
   pip install PyMuPDF
   pip install transformers
   pip install torch
   
   # Dependencias adicionales
   pip install numpy
   pip install pillow
   pip install requests
   ```

### Verificación de la Instalación

Ejecuta este comando para verificar que todo está correcto:
```bash
python sistema_ocr_avanzado.py --help
```

Deberías ver la ayuda del sistema sin errores.

## 🚀 Uso

### Preparación
1. **Coloca tus documentos** en la carpeta `entrada/`
2. **Asegúrate de que el entorno virtual esté activado**:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

### Ejecución

#### Modo Rápido (por defecto)
```bash
# Windows
python sistema_ocr_avanzado.py --modo rapido

# macOS/Linux
python3 sistema_ocr_avanzado.py --modo rapido
```

#### Modo Preciso
```bash
# Windows
python sistema_ocr_avanzado.py --modo preciso

# macOS/Linux
python3 sistema_ocr_avanzado.py --modo preciso
```

### Formatos de Documentos Soportados
- **PDF**: `.pdf` (primera página en modo rápido)
- **Imágenes**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`

### Estructura de Carpetas
El sistema crea automáticamente estas carpetas:
- `entrada/` - Documentos a procesar
- `procesados/` - Documentos ya procesados
- `resultados/` - Resultados JSON
- `logs/` - Logs del sistema

## 📁 Estructura del Proyecto

```
OCR/
├── sistema_ocr_avanzado.py    # Script principal
├── configuracion/             # Archivos de configuración
│   ├── facturas.txt
│   ├── recibos.txt
│   ├── multas.txt
│   ├── contratos.txt
│   └── otros.txt
├── entrada/                   # Documentos a procesar
├── procesados/               # Documentos procesados
├── resultados/               # Resultados JSON
├── logs/                     # Logs del sistema
└── README.md
```

## 📊 Salida Unificada

Todos los documentos generan la misma estructura JSON:

```json
{
  "cabecera": {
    "nif_emisor": "A12345678",
    "fecha_emision": "30/07/2020",
    "razon_social_emisor": "EMPRESA, S.L.",
    "numero_documento": "FAC-001",
    "tipo_documento": "FACTURA"
  },
  "lineas": [
    {
      "numero_linea": 1,
      "descripcion": "Producto/Servicio",
      "cantidad": "1",
      "precio_unitario": "100,00",
      "importe_linea": "100,00",
      "accion": "COMPRA",
      "confianza": 0.95
    }
  ],
  "totales": {
    "base_imponible": "82,64",
    "iva": {
      "21%": "17,36"
    },
    "total": "100,00",
    "moneda": "EUR"
  },
  "metadatos": {
    "archivo_original": "documento.pdf",
    "confianza_ocr": 0.99,
    "confianza_clasificacion": 0.95,
    "confianza_validacion": 0.98,
    "confianza_final": 0.97,
    "requiere_revision": false,
    "timestamp_procesamiento": "2025-09-28T18:00:00Z"
  }
}
```

## 🔧 Configuración

### Modo Rápido
- Desactiva Phi-3-mini
- Preprocesamiento mínimo
- Warm-up OCR optimizado
- Procesamiento PDF primera página

### Modo Preciso
- Phi-3-mini activo si confianza < 0.98
- Preprocesamiento completo
- Validación exhaustiva
- Corrección contextual

## 📈 Rendimiento

- **Modo Rápido**: < 3 segundos/documento
- **Modo Preciso**: < 10 segundos/documento
- **Precisión**: 99% en campos clave
- **Soporte**: Lotes de 100+ documentos

## 🏢 Empresas Soportadas

- **Supermercados**: Mercadona, Carrefour, El Corte Inglés, Lidl, Aldi
- **Aerolíneas**: AirEuropa, Iberia, Vueling, Ryanair, EasyJet
- **Bancos**: BBVA, Santander, CaixaBank, Sabadell
- **Talleres**: Automatri, Mogacar
- **Otros**: ICASSO, restaurantes, bares, etc.

## 📝 Logs

El sistema genera logs detallados en `logs/` con:
- Timestamp de procesamiento
- Estadísticas de rendimiento
- Errores y advertencias
- Resumen de resultados

## 🔍 Validación

- **Confianza OCR**: Calidad del reconocimiento de texto
- **Confianza Clasificación**: Precisión del tipo de documento
- **Confianza Validación**: Verificación de campos extraídos
- **Confianza Final**: Puntuación global del procesamiento

## 🚨 Requiere Revisión

Los documentos con `confianza_final < 0.95` se marcan para revisión humana.

## 🔧 Solución de Problemas

### Errores Comunes

#### "ModuleNotFoundError: No module named 'cv2'"
```bash
pip install opencv-python
```

#### "ModuleNotFoundError: No module named 'fitz'"
```bash
pip install PyMuPDF
```

#### "ModuleNotFoundError: No module named 'paddle'"
```bash
pip install paddlepaddle
```

#### Error de permisos en Windows
- Ejecuta PowerShell como administrador
- O usa: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

#### Error de memoria en macOS
```bash
# Aumentar límite de memoria
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

### Verificación del Sistema
```bash
# Verificar Python
python --version

# Verificar dependencias
python -c "import cv2, fitz, paddle, transformers, torch; print('Todas las dependencias OK')"

# Verificar el sistema OCR
python sistema_ocr_avanzado.py --help
```

## 📞 Soporte

- **Issues**: [Crear un issue](https://github.com/armandix23/sistema-ocr-avanzado/issues)
- **Documentación**: Ver este README
- **Ejemplos**: Revisa la carpeta `configuracion/` para patrones

## 📄 Licencia

Este proyecto está bajo la Licencia Apache 2.0.

---

**Versión**: 2.0 Ultra-Rápida  
**Compatibilidad**: Windows 10/11, macOS 10.15+, Linux Ubuntu 20.04+  
**Python**: 3.12+  
**Última actualización**: Septiembre 2025