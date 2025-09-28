#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de prueba completo para la API OCR
Prueba todos los endpoints y funcionalidades
"""

import requests
import json
import time
from pathlib import Path

# Configuración
API_BASE_URL = "http://localhost:8000"

def test_health():
    """Probar endpoint de salud"""
    print("🔍 Probando endpoint de salud...")
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()

def test_root():
    """Probar endpoint raíz"""
    print("🏠 Probando endpoint raíz...")
    response = requests.get(f"{API_BASE_URL}/")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Message: {data.get('message')}")
    print(f"Version: {data.get('version')}")
    print(f"Endpoints: {len(data.get('endpoints', {}))}")
    print()

def test_status():
    """Probar endpoint de estado"""
    print("📊 Probando endpoint de estado...")
    response = requests.get(f"{API_BASE_URL}/status")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()

def test_upload(file_path):
    """Probar subida de archivo"""
    print(f"📤 Probando subida de archivo: {file_path}")
    
    if not Path(file_path).exists():
        print(f"❌ Archivo no encontrado: {file_path}")
        return None
    
    with open(file_path, 'rb') as f:
        files = {'file': (Path(file_path).name, f, 'application/pdf')}
        response = requests.post(f"{API_BASE_URL}/upload", files=files)
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Archivo procesado exitosamente:")
        print(f"  - Filename: {result.get('filename')}")
        print(f"  - Tipo: {result.get('tipo_documento')}")
        print(f"  - Confianza: {result.get('confianza_final', 0):.3f}")
        print(f"  - Requiere revisión: {result.get('requiere_revision', False)}")
        
        # Mostrar estructura de resultado
        resultado = result.get('resultado', {})
        if resultado:
            cabecera = resultado.get('cabecera', {})
            totales = resultado.get('totales', {})
            metadatos = resultado.get('metadatos', {})
            
            print("  - Estructura del resultado:")
            print(f"    * Empresa: {cabecera.get('razon_social_emisor', 'N/A')}")
            print(f"    * Fecha: {cabecera.get('fecha_emision', 'N/A')}")
            print(f"    * Total: {totales.get('total', 'N/A')}")
            print(f"    * Líneas: {len(resultado.get('lineas', []))}")
            print(f"    * Confianza final: {metadatos.get('confianza_final', 0):.3f}")
        
        return result.get('filename')
    else:
        print(f"❌ Error: {response.json()}")
        return None

def test_get_result(filename):
    """Probar obtención de resultado"""
    print(f"📥 Probando obtención de resultado: {filename}")
    
    response = requests.get(f"{API_BASE_URL}/result/{filename}")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Resultado obtenido:")
        print(f"  - Filename: {result.get('filename')}")
        print(f"  - Tipo: {result.get('tipo_documento')}")
        print(f"  - Confianza: {result.get('confianza_final', 0):.3f}")
        print(f"  - Requiere revisión: {result.get('requiere_revision', False)}")
        
        # Mostrar estructura completa
        resultado = result.get('resultado', {})
        if resultado:
            print("  - Estructura completa:")
            print(json.dumps(resultado, indent=4, ensure_ascii=False))
    else:
        print(f"❌ Error: {response.json()}")
    print()

def test_list_results():
    """Probar listado de resultados"""
    print("📋 Probando listado de resultados...")
    response = requests.get(f"{API_BASE_URL}/results")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Total resultados: {result.get('total', 0)}")
        
        resultados = result.get('resultados', [])
        for i, res in enumerate(resultados[:3]):  # Mostrar solo los primeros 3
            print(f"  {i+1}. {res.get('filename')} - {res.get('tipo_documento')} - {res.get('confianza_final', 0):.3f}")
        
        if len(resultados) > 3:
            print(f"  ... y {len(resultados) - 3} más")
    else:
        print(f"❌ Error: {response.json()}")
    print()

def test_error_cases():
    """Probar casos de error"""
    print("🚨 Probando casos de error...")
    
    # Archivo no soportado
    print("  - Probando archivo no soportado...")
    with open("test.txt", "w") as f:
        f.write("test")
    
    with open("test.txt", "rb") as f:
        files = {'file': ('test.txt', f, 'text/plain')}
        response = requests.post(f"{API_BASE_URL}/upload", files=files)
    
    print(f"    Status: {response.status_code} (esperado: 400)")
    
    # Limpiar archivo temporal
    Path("test.txt").unlink(missing_ok=True)
    
    # Resultado no encontrado
    print("  - Probando resultado no encontrado...")
    response = requests.get(f"{API_BASE_URL}/result/archivo_inexistente")
    print(f"    Status: {response.status_code} (esperado: 404)")
    print()

def main():
    """Función principal de prueba"""
    print("🚀 Iniciando pruebas completas de la API OCR")
    print("=" * 60)
    
    # Probar endpoints básicos
    test_health()
    test_root()
    test_status()
    
    # Probar subida de archivo (usar un archivo de prueba si existe)
    test_files = [
        "entrada/test.pdf",
        "entrada/factura.pdf", 
        "entrada/recibo.pdf",
        "entrada/multa.pdf"
    ]
    
    filename = None
    for test_file in test_files:
        if Path(test_file).exists():
            filename = test_upload(test_file)
            break
    
    if not filename:
        print("⚠️  No se encontraron archivos de prueba en la carpeta 'entrada'")
        print("   Crea un archivo PDF, JPG o PNG en 'entrada/' para probar la funcionalidad completa")
    else:
        # Probar obtención de resultado
        test_get_result(filename)
    
    # Probar listado de resultados
    test_list_results()
    
    # Probar casos de error
    test_error_cases()
    
    print("✅ Pruebas completadas")
    print("\n📚 Documentación disponible en: http://localhost:8000/docs")

if __name__ == "__main__":
    main()
