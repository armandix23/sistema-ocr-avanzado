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

### Requisitos
- Python 3.12+
- Windows 10/11
- 4GB RAM mínimo
- 2GB espacio en disco

### Dependencias
```bash
pip install paddlepaddle
pip install opencv-python
pip install PyMuPDF
pip install transformers
pip install torch
```

## 🚀 Uso

### Modo Rápido (por defecto)
```bash
python sistema_ocr_avanzado.py --modo rapido
```

### Modo Preciso
```bash
python sistema_ocr_avanzado.py --modo preciso
```

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

## 📞 Soporte

Para problemas o mejoras, crear un issue en el repositorio.

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.

---

**Versión**: 2.0 Ultra-Rápida  
**Compatibilidad**: Windows + Python 3.12  
**Última actualización**: Septiembre 2025