#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de prueba para la API OCR
"""

import requests
import json
import time
from pathlib import Path

# Configuración
API_BASE_URL = "http://localhost:8000"
TEST_FILE = "test_document.pdf"  # Cambia por un archivo real

def test_health():
    """Probar endpoint de salud"""
    print("🔍 Probando endpoint de salud...")
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_status():
    """Probar endpoint de estado"""
    print("📊 Probando endpoint de estado...")
    response = requests.get(f"{API_BASE_URL}/status")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
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
    result = response.json()
    print(f"Response: {result}")
    print()
    
    return result.get('filename') if response.status_code == 200 else None

def test_get_result(filename):
    """Probar obtención de resultado"""
    print(f"📥 Probando obtención de resultado: {filename}")
    
    # Esperar un poco para que se procese
    print("⏳ Esperando procesamiento...")
    time.sleep(5)
    
    response = requests.get(f"{API_BASE_URL}/result/{filename}")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Resultado obtenido:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"❌ Error: {response.json()}")
    print()

def test_list_results():
    """Probar listado de resultados"""
    print("📋 Probando listado de resultados...")
    response = requests.get(f"{API_BASE_URL}/results")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def main():
    """Función principal de prueba"""
    print("🚀 Iniciando pruebas de la API OCR")
    print("=" * 50)
    
    # Probar endpoints básicos
    test_health()
    test_status()
    
    # Probar subida de archivo
    filename = test_upload(TEST_FILE)
    
    if filename:
        # Probar obtención de resultado
        test_get_result(filename)
    
    # Probar listado de resultados
    test_list_results()
    
    print("✅ Pruebas completadas")

if __name__ == "__main__":
    main()


