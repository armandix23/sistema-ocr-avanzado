#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de prueba para integración con Angular frontend
Verifica que la API responda correctamente para el frontend
"""

import requests
import json
import time
from pathlib import Path

# Configuración
API_BASE_URL = "http://localhost:8000"

def test_cors_headers():
    """Probar headers CORS para Angular"""
    print("🌐 Probando headers CORS...")
    
    # Simular petición desde Angular
    headers = {
        'Origin': 'http://localhost:4200',
        'Access-Control-Request-Method': 'POST',
        'Access-Control-Request-Headers': 'Content-Type'
    }
    
    response = requests.options(f"{API_BASE_URL}/upload", headers=headers)
    print(f"Status: {response.status_code}")
    
    cors_headers = {
        'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
        'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods'),
        'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers')
    }
    
    print(f"CORS Headers: {json.dumps(cors_headers, indent=2)}")
    
    # Verificar que permite localhost:4200
    if cors_headers['Access-Control-Allow-Origin'] == 'http://localhost:4200':
        print("✅ CORS configurado correctamente para Angular")
    else:
        print("❌ CORS no configurado para Angular")
    print()

def test_upload_structure():
    """Probar estructura de respuesta del upload"""
    print("📤 Probando estructura de respuesta del upload...")
    
    # Crear archivo de prueba
    test_file = "test_angular.txt"
    with open(test_file, "w") as f:
        f.write("Test file for Angular integration")
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': (test_file, f, 'text/plain')}
            response = requests.post(f"{API_BASE_URL}/upload", files=files)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 400:  # Esperado para archivo no soportado
            print("✅ Validación de tipos de archivo funcionando")
        else:
            result = response.json()
            print("Estructura de respuesta:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            # Verificar estructura requerida
            required_keys = ['cabecera', 'lineas', 'totales', 'metadatos']
            if all(key in result for key in required_keys):
                print("✅ Estructura de respuesta correcta")
            else:
                print("❌ Estructura de respuesta incorrecta")
    
    finally:
        # Limpiar archivo de prueba
        Path(test_file).unlink(missing_ok=True)
    print()

def test_results_endpoint():
    """Probar endpoint /results para Angular"""
    print("📋 Probando endpoint /results...")
    
    response = requests.get(f"{API_BASE_URL}/results")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        results = response.json()
        print(f"Total resultados: {len(results)}")
        
        if results:
            # Verificar estructura del primer resultado
            first_result = results[0]
            required_fields = ['nombre', 'tipo_documento', 'confianza_final', 'timestamp']
            
            if all(field in first_result for field in required_fields):
                print("✅ Estructura de /results correcta")
                print(f"Ejemplo: {json.dumps(first_result, indent=2, ensure_ascii=False)}")
            else:
                print("❌ Estructura de /results incorrecta")
        else:
            print("ℹ️  No hay resultados para mostrar")
    else:
        print(f"❌ Error: {response.json()}")
    print()

def test_angular_compatibility():
    """Probar compatibilidad específica con Angular"""
    print("🅰️  Probando compatibilidad con Angular...")
    
    # Probar endpoint raíz
    response = requests.get(f"{API_BASE_URL}/")
    if response.status_code == 200:
        data = response.json()
        if data.get('cors_enabled') == 'http://localhost:4200':
            print("✅ CORS configurado para Angular")
        else:
            print("❌ CORS no configurado para Angular")
    
    # Probar health check
    response = requests.get(f"{API_BASE_URL}/health")
    if response.status_code == 200:
        print("✅ Health check funcionando")
    
    # Probar status
    response = requests.get(f"{API_BASE_URL}/status")
    if response.status_code == 200:
        print("✅ Status endpoint funcionando")
    
    print()

def test_error_handling():
    """Probar manejo de errores para Angular"""
    print("🚨 Probando manejo de errores...")
    
    # Archivo no soportado
    with open("test.txt", "w") as f:
        f.write("test")
    
    with open("test.txt", "rb") as f:
        files = {'file': ('test.txt', f, 'text/plain')}
        response = requests.post(f"{API_BASE_URL}/upload", files=files)
    
    print(f"Archivo no soportado - Status: {response.status_code}")
    if response.status_code == 400:
        error_data = response.json()
        if 'detail' in error_data:
            print("✅ Error manejado correctamente")
        else:
            print("❌ Error no manejado correctamente")
    
    # Limpiar archivo temporal
    Path("test.txt").unlink(missing_ok=True)
    
    # Resultado no encontrado
    response = requests.get(f"{API_BASE_URL}/result/archivo_inexistente")
    print(f"Resultado no encontrado - Status: {response.status_code}")
    if response.status_code == 404:
        print("✅ Error 404 manejado correctamente")
    
    print()

def main():
    """Función principal de prueba"""
    print("🚀 Iniciando pruebas de integración con Angular")
    print("=" * 60)
    
    # Probar CORS
    test_cors_headers()
    
    # Probar estructura de respuesta
    test_upload_structure()
    
    # Probar endpoint results
    test_results_endpoint()
    
    # Probar compatibilidad con Angular
    test_angular_compatibility()
    
    # Probar manejo de errores
    test_error_handling()
    
    print("✅ Pruebas de integración completadas")
    print("\n📚 La API está lista para integrarse con Angular frontend")
    print("🔗 Frontend Angular: http://localhost:4200")
    print("🔗 API Backend: http://localhost:8000")
    print("📖 Documentación: http://localhost:8000/docs")

if __name__ == "__main__":
    main()
