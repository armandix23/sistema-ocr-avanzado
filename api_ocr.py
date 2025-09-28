#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API REST para Sistema OCR Ultra-Rápido
FastAPI + uvicorn para procesamiento de documentos
"""

import os
import sys
import json
import shutil
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from queue import Queue
import threading

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Importar el sistema OCR existente
from sistema_ocr_avanzado import (
    procesar_archivo,
    mover_archivo_procesado,
    guardar_resultado,
    hacer_warmup_ocr,
    inicializar_phi3,
    clasificar_documento_semantico,
    extraer_campos_semanticos,
    validar_extraccion_semantica,
    generar_salida_unificada,
    crear_estructura_carpetas,
    verificar_dependencias
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="Sistema OCR Ultra-Rápido API",
    description="API REST para procesamiento OCR de documentos con clasificación automática",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cola de procesamiento (en memoria)
processing_queue = Queue()
processing_lock = threading.Lock()
is_processing = False

# Variables globales para OCR
ocr_engine = None
tokenizer_phi3 = None
model_phi3 = None

class ProcessingStatus:
    """Estado del procesamiento"""
    def __init__(self):
        self.status = "idle"
        self.current_file = None
        self.progress = 0
        self.error = None

processing_status = ProcessingStatus()

@app.on_event("startup")
async def startup_event():
    """Inicialización de la API"""
    global ocr_engine, tokenizer_phi3, model_phi3
    
    logger.info("Iniciando Sistema OCR Ultra-Rápido API...")
    
    # Verificar dependencias
    if not verificar_dependencias():
        logger.error("Dependencias no encontradas")
        return
    
    # Crear estructura de carpetas
    crear_estructura_carpetas()
    
    # Inicializar OCR
    try:
        from paddleocr import PaddleOCR
        ocr_engine = PaddleOCR(use_angle_cls=True, lang='es', show_log=False)
        hacer_warmup_ocr(ocr_engine)
        logger.info("OCR engine inicializado correctamente")
    except Exception as e:
        logger.error(f"Error inicializando OCR: {e}")
        ocr_engine = None
    
    # Inicializar Phi-3 (modo rápido por defecto)
    tokenizer_phi3, model_phi3 = inicializar_phi3(modo_rapido=True)
    
    logger.info("API lista para procesar documentos")

@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "message": "Sistema OCR Ultra-Rápido API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "upload": "/upload - Subir archivo y obtener resultado inmediato",
            "result": "/result/{filename} - Obtener resultado específico",
            "results": "/results - Listar todos los resultados",
            "status": "/status - Estado del procesamiento",
            "health": "/health - Salud de la API"
        },
        "formats_supported": ["PDF", "JPG", "JPEG", "PNG", "BMP", "TIFF"],
        "document_types": ["FACTURA", "RECIBO", "MULTA", "CONTRATO", "OTROS"],
        "response_structure": {
            "cabecera": "Información del emisor y documento",
            "lineas": "Productos o conceptos",
            "totales": "Base imponible, IVA y total",
            "metadatos": "Confianza, validación y timestamp"
        }
    }

@app.get("/health")
async def health_check():
    """Verificación de salud de la API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ocr_engine": "ready" if ocr_engine else "not_ready",
        "processing_queue": processing_queue.qsize()
    }

@app.get("/status")
async def get_status():
    """Estado actual del procesamiento"""
    return {
        "status": processing_status.status,
        "current_file": processing_status.current_file,
        "progress": processing_status.progress,
        "error": processing_status.error,
        "queue_size": processing_queue.qsize(),
        "timestamp": datetime.now().isoformat()
    }

def process_document(file_path: str, filename: str) -> Dict[str, Any]:
    """Procesa un documento con el sistema OCR"""
    global processing_status
    
    try:
        processing_status.status = "processing"
        processing_status.current_file = filename
        processing_status.progress = 10
        processing_status.error = None
        
        # Procesar archivo
        resultado = procesar_archivo(
            file_path, 
            ocr_engine, 
            modo_rapido=True, 
            tokenizer_phi3=tokenizer_phi3, 
            model_phi3=model_phi3
        )
        
        processing_status.progress = 80
        
        # Extraer información del resultado
        tipo_documento = resultado.get('clasificacion', {}).get('tipo_detectado', 'OTROS')
        campos_extraidos = resultado.get('campos_extraidos', {})
        estadisticas = resultado.get('estadisticas', {})
        
        # Generar salida unificada
        salida_unificada = generar_salida_unificada(
            campos_extraidos, 
            tipo_documento, 
            estadisticas, 
            filename
        )
        
        processing_status.progress = 90
        
        # Guardar resultado
        archivo_resultado = guardar_resultado(resultado, filename, tipo_documento)
        
        # Mover archivo procesado
        archivo_procesado = mover_archivo_procesado(Path(file_path), tipo_documento)
        
        processing_status.progress = 100
        processing_status.status = "completed"
        
        return {
            "success": True,
            "resultado": salida_unificada,
            "archivo_resultado": archivo_resultado,
            "archivo_procesado": str(archivo_procesado),
            "tipo_documento": tipo_documento,
            "confianza_final": estadisticas.get('confianza_final', 0.0)
        }
        
    except Exception as e:
        processing_status.status = "error"
        processing_status.error = str(e)
        logger.error(f"Error procesando {filename}: {e}")
        return {
            "success": False,
            "error": str(e),
            "filename": filename
        }

def worker():
    """Worker para procesar archivos de la cola"""
    global is_processing, processing_status
    
    while True:
        try:
            if not processing_queue.empty():
                with processing_lock:
                    if is_processing:
                        continue
                    is_processing = True
                
                # Obtener archivo de la cola
                file_info = processing_queue.get()
                file_path = file_info['file_path']
                filename = file_info['filename']
                
                logger.info(f"Procesando archivo: {filename}")
                
                # Procesar documento
                resultado = process_document(file_path, filename)
                
                # Marcar tarea como completada
                processing_queue.task_done()
                
                with processing_lock:
                    is_processing = False
                
                logger.info(f"Archivo procesado: {filename}")
                
            else:
                # No hay archivos en la cola, esperar
                asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error en worker: {e}")
            with processing_lock:
                is_processing = False

# Iniciar worker en hilo separado
worker_thread = threading.Thread(target=worker, daemon=True)
worker_thread.start()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Subir y procesar un archivo con respuesta inmediata"""
    global processing_status
    
    # Validar tipo de archivo
    allowed_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no soportado. Extensiones permitidas: {', '.join(allowed_extensions)}"
        )
    
    # Verificar que no se esté procesando otro archivo
    if processing_status.status == "processing":
        raise HTTPException(
            status_code=429,
            detail="Sistema ocupado procesando otro archivo. Intenta más tarde."
        )
    
    try:
        # Crear nombre único para el archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{Path(file.filename).stem}_{timestamp}{file_extension}"
        file_path = Path("entrada") / filename
        
        # Guardar archivo
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"Archivo guardado: {filename}")
        
        # Procesar inmediatamente (sin cola para respuesta directa)
        processing_status.status = "processing"
        processing_status.current_file = filename
        processing_status.progress = 10
        processing_status.error = None
        
        # Procesar archivo
        resultado = procesar_archivo(
            str(file_path), 
            ocr_engine, 
            modo_rapido=True, 
            tokenizer_phi3=tokenizer_phi3, 
            model_phi3=model_phi3
        )
        
        processing_status.progress = 80
        
        # Extraer información del resultado
        tipo_documento = resultado.get('clasificacion', {}).get('tipo_detectado', 'OTROS')
        campos_extraidos = resultado.get('campos_extraidos', {})
        estadisticas = resultado.get('estadisticas', {})
        
        # Generar salida unificada
        salida_unificada = generar_salida_unificada(
            campos_extraidos, 
            tipo_documento, 
            estadisticas, 
            filename
        )
        
        processing_status.progress = 90
        
        # Guardar resultado
        archivo_resultado = guardar_resultado(resultado, filename, tipo_documento)
        
        # Mover archivo procesado
        archivo_procesado = mover_archivo_procesado(Path(file_path), tipo_documento)
        
        processing_status.progress = 100
        processing_status.status = "completed"
        
        # Devolver resultado inmediatamente
        return {
            "success": True,
            "message": "Archivo procesado correctamente",
            "filename": filename,
            "resultado": salida_unificada,
            "tipo_documento": tipo_documento,
            "confianza_final": estadisticas.get('confianza_final', 0.0),
            "requiere_revision": estadisticas.get('requiere_revision', False),
            "archivo_resultado": archivo_resultado,
            "archivo_procesado": str(archivo_procesado)
        }
        
    except Exception as e:
        processing_status.status = "error"
        processing_status.error = str(e)
        logger.error(f"Error procesando archivo: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando archivo: {str(e)}")

@app.get("/result/{filename}")
async def get_result(filename: str):
    """Obtener resultado de un archivo procesado"""
    try:
        # Buscar archivo de resultado
        resultados_dir = Path("resultados")
        for subdir in resultados_dir.iterdir():
            if subdir.is_dir():
                result_file = subdir / f"{filename}.json"
                if result_file.exists():
                    with open(result_file, 'r', encoding='utf-8') as f:
                        resultado = json.load(f)
                    
                    # Extraer salida unificada
                    salida_unificada = resultado.get('salida_unificada', {})
                    metadatos = salida_unificada.get('metadatos', {})
                    
                    return {
                        "success": True,
                        "filename": filename,
                        "resultado": salida_unificada,
                        "tipo_documento": salida_unificada.get('cabecera', {}).get('tipo_documento'),
                        "confianza_final": metadatos.get('confianza_final', 0.0),
                        "requiere_revision": metadatos.get('requiere_revision', False),
                        "timestamp": metadatos.get('timestamp_procesamiento')
                    }
        
        raise HTTPException(status_code=404, detail="Resultado no encontrado")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo resultado: {e}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo resultado: {str(e)}")

@app.get("/results")
async def list_results():
    """Listar todos los resultados disponibles"""
    try:
        resultados = []
        resultados_dir = Path("resultados")
        
        if resultados_dir.exists():
            for subdir in resultados_dir.iterdir():
                if subdir.is_dir():
                    for result_file in subdir.glob("*.json"):
                        with open(result_file, 'r', encoding='utf-8') as f:
                            resultado = json.load(f)
                        
                        salida_unificada = resultado.get('salida_unificada', {})
                        metadatos = salida_unificada.get('metadatos', {})
                        cabecera = salida_unificada.get('cabecera', {})
                        totales = salida_unificada.get('totales', {})
                        
                        resultados.append({
                            "filename": result_file.stem,
                            "tipo_documento": cabecera.get('tipo_documento'),
                            "empresa_emisor": cabecera.get('razon_social_emisor'),
                            "fecha_emision": cabecera.get('fecha_emision'),
                            "total": totales.get('total'),
                            "confianza_final": metadatos.get('confianza_final', 0.0),
                            "requiere_revision": metadatos.get('requiere_revision', False),
                            "timestamp": metadatos.get('timestamp_procesamiento'),
                            "archivo_original": metadatos.get('archivo_original')
                        })
        
        return {
            "success": True,
            "total": len(resultados),
            "resultados": resultados
        }
        
    except Exception as e:
        logger.error(f"Error listando resultados: {e}")
        raise HTTPException(status_code=500, detail=f"Error listando resultados: {str(e)}")

@app.delete("/result/{filename}")
async def delete_result(filename: str):
    """Eliminar resultado de un archivo"""
    try:
        # Buscar y eliminar archivo de resultado
        resultados_dir = Path("resultados")
        deleted = False
        
        for subdir in resultados_dir.iterdir():
            if subdir.is_dir():
                result_file = subdir / f"{filename}.json"
                if result_file.exists():
                    result_file.unlink()
                    deleted = True
                    break
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Resultado no encontrado")
        
        return {
            "success": True,
            "message": f"Resultado {filename} eliminado correctamente"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error eliminando resultado: {e}")
        raise HTTPException(status_code=500, detail=f"Error eliminando resultado: {str(e)}")

if __name__ == "__main__":
    # Configuración para desarrollo
    uvicorn.run(
        "api_ocr:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


