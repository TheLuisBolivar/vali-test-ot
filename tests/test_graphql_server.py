"""
Tests para el servidor GraphQL del Sistema de Optimización de Proceso Industrial.

Estos tests verifican:
1. La correcta carga de datos desde archivos JSON
2. La ejecución adecuada de queries GraphQL para obtener datos
3. La ejecución adecuada de mutations para insertar nuevos datos
"""

import unittest
import os
import json
import pandas as pd
import tempfile
import shutil
import pytest
from datetime import datetime
from src.graphql_server import (
    load_process_data, 
    load_optimization_results, 
    load_analysis_results,
    append_process_data,
    save_optimization_results,
    ProcessVariableInput,
    RESULTS_DIR,
    Query,
    Mutation
)

class Environment:
    """Clase para gestionar el entorno de pruebas aislado."""
    
    def __init__(self):
        """Inicializa un entorno temporal para pruebas."""
        # Guardar referencia al directorio original
        self.original_dir = os.environ.get("RESULTS_DIR", RESULTS_DIR)
        # Crear directorio temporal
        self.temp_dir = tempfile.mkdtemp(prefix="test_graphql_")
        
        # Guardar copias de los archivos originales si existen
        self.has_original_process_data = False
        self.has_original_optimization_results = False
        
        # Comprobar y copiar datos de proceso
        if os.path.exists(f"{self.original_dir}/process_data.json"):
            self.has_original_process_data = True
            with open(f"{self.original_dir}/process_data.json", "r") as src:
                os.makedirs(self.temp_dir, exist_ok=True)
                with open(f"{self.temp_dir}/process_data.json", "w") as dst:
                    dst.write(src.read())
        
        # Comprobar y copiar resultados de optimización
        if os.path.exists(f"{self.original_dir}/optimization_results.json"):
            self.has_original_optimization_results = True
            with open(f"{self.original_dir}/optimization_results.json", "r") as src:
                os.makedirs(self.temp_dir, exist_ok=True)
                with open(f"{self.temp_dir}/optimization_results.json", "w") as dst:
                    dst.write(src.read())
    
    def setup(self):
        """Activa el entorno de pruebas."""
        os.environ["RESULTS_DIR"] = self.temp_dir
    
    def teardown(self):
        """Restaura el entorno original."""
        os.environ["RESULTS_DIR"] = self.original_dir
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def __enter__(self):
        """Soporte para context manager."""
        self.setup()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Limpieza al salir del context manager."""
        self.teardown()

class TestGraphQLServer(unittest.TestCase):
    """Clase de pruebas para el servidor GraphQL."""
    
    def setUp(self):
        """Configura el entorno de pruebas."""
        # Crear entorno de pruebas
        self.env = Environment()
        self.env.setup()
    
    def tearDown(self):
        """Limpia el entorno después de las pruebas."""
        self.env.teardown()
    
    def test_load_process_data(self):
        """Prueba la carga de datos de proceso."""
        # Si no hay datos originales, crear datos de prueba
        if not self.env.has_original_process_data:
            test_data = [
                {"timestamp": "2023-10-01T08:00:00", "temperature": 85.3, "pressure": 102.4, "velocity": 45.1, "humidity": 65.2, "qualityScore": 87.3},
                {"timestamp": "2023-10-01T09:00:00", "temperature": 86.1, "pressure": 103.1, "velocity": 46.0, "humidity": 64.8, "qualityScore": 88.1}
            ]
            
            os.makedirs(self.env.temp_dir, exist_ok=True)
            with open(f"{self.env.temp_dir}/process_data.json", "w") as f:
                json.dump(test_data, f)
        
        # Cargar datos
        data = load_process_data()
        
        # Verificar que los datos se cargaron correctamente
        self.assertIsNotNone(data, "No se pudieron cargar los datos de proceso")
        self.assertGreater(len(data), 0, "No se encontraron datos de proceso")
        
        # Verificar que los objetos tienen la estructura esperada
        sample = data[0]
        self.assertTrue(hasattr(sample, "timestamp"), "Falta el atributo timestamp")
        self.assertTrue(hasattr(sample, "temperature"), "Falta el atributo temperature")
        self.assertTrue(hasattr(sample, "pressure"), "Falta el atributo pressure")
        self.assertTrue(hasattr(sample, "velocity"), "Falta el atributo velocity")
        self.assertTrue(hasattr(sample, "humidity"), "Falta el atributo humidity")
        self.assertTrue(hasattr(sample, "qualityScore"), "Falta el atributo qualityScore")
    
    def test_load_optimization_results(self):
        """Prueba la carga de resultados de optimización."""
        # Si no hay datos originales, crear datos de prueba
        if not self.env.has_original_optimization_results:
            test_data = {
                "optimalParameters": {
                    "temperature": 88.5,
                    "pressure": 103.2,
                    "velocity": 45.0,
                    "humidity": 63.0
                },
                "predictedQuality": 90.5,
                "confidenceScore": 0.95,
                "optimizationTime": 2.5
            }
            
            os.makedirs(self.env.temp_dir, exist_ok=True)
            with open(f"{self.env.temp_dir}/optimization_results.json", "w") as f:
                json.dump(test_data, f)
        
        # Cargar resultados
        results = load_optimization_results()
        
        # Verificar que los resultados se cargaron correctamente
        self.assertIsNotNone(results, "No se pudieron cargar los resultados de optimización")
        
        # Verificar que los objetos tienen la estructura esperada
        self.assertTrue(hasattr(results, "optimalParameters"), "Falta el atributo optimalParameters")
        self.assertTrue(hasattr(results, "predictedQuality"), "Falta el atributo predictedQuality")
        self.assertTrue(hasattr(results, "confidenceScore"), "Falta el atributo confidenceScore")
        self.assertTrue(hasattr(results, "optimizationTime"), "Falta el atributo optimizationTime")
    
    def test_query_process_data(self):
        """Prueba el Query para obtener datos de proceso."""
        # Si no hay datos originales, crear datos de prueba
        if not self.env.has_original_process_data:
            test_data = [
                {"timestamp": "2023-10-01T08:00:00", "temperature": 85.3, "pressure": 102.4, "velocity": 45.1, "humidity": 65.2, "qualityScore": 87.3},
                {"timestamp": "2023-10-01T09:00:00", "temperature": 86.1, "pressure": 103.1, "velocity": 46.0, "humidity": 64.8, "qualityScore": 88.1}
            ]
            
            os.makedirs(self.env.temp_dir, exist_ok=True)
            with open(f"{self.env.temp_dir}/process_data.json", "w") as f:
                json.dump(test_data, f)
        
        # Crear instancia de Query
        query = Query()
        
        # Ejecutar query para obtener todos los datos
        all_data = query.process_data()
        self.assertIsNotNone(all_data, "El query no devolvió datos")
        self.assertGreater(len(all_data), 0, "No se encontraron datos de proceso")
        
        # Ejecutar query para obtener datos recientes
        latest_data = query.latest_process_data(count=2)
        self.assertIsNotNone(latest_data, "El query no devolvió datos recientes")
        self.assertLessEqual(len(latest_data), 2, "El query devolvió más datos de los solicitados")
    
    def test_query_optimization_results(self):
        """Prueba el Query para obtener resultados de optimización."""
        # Si no hay datos originales, crear datos de prueba
        if not self.env.has_original_optimization_results:
            test_data = {
                "optimalParameters": {
                    "temperature": 88.5,
                    "pressure": 103.2,
                    "velocity": 45.0,
                    "humidity": 63.0
                },
                "predictedQuality": 90.5,
                "confidenceScore": 0.95,
                "optimizationTime": 2.5
            }
            
            os.makedirs(self.env.temp_dir, exist_ok=True)
            with open(f"{self.env.temp_dir}/optimization_results.json", "w") as f:
                json.dump(test_data, f)
        
        # Crear instancia de Query
        query = Query()
        
        # Ejecutar query para obtener resultados de optimización
        results = query.optimization_results()
        
        # Verificar resultados
        self.assertIsNotNone(results, "El query no devolvió resultados de optimización")
        self.assertTrue(hasattr(results, "optimalParameters"), "Faltan los parámetros óptimos en los resultados")
        self.assertTrue(hasattr(results, "predictedQuality"), "Falta la calidad predicha en los resultados")
    
    def test_process_data_query_count(self):
        """Prueba que el query para obtener todos los datos de proceso devuelva 31 elementos."""
        # Crear datos de prueba con 31 elementos
        test_data = []
        for i in range(31):
            test_data.append({
                "timestamp": f"2023-10-01T{i:02d}:00:00",
                "temperature": 85.0 + (i * 0.1),
                "pressure": 102.0 + (i * 0.1),
                "velocity": 45.0 + (i * 0.1),
                "humidity": 65.0 - (i * 0.1),
                "qualityScore": 87.0 + (i * 0.1)
            })
        
        # Guardar datos de prueba
        os.makedirs(self.env.temp_dir, exist_ok=True)
        with open(f"{self.env.temp_dir}/process_data.json", "w") as f:
            json.dump(test_data, f)
        
        # Crear instancia de Query
        query = Query()
        
        # Ejecutar query para obtener todos los datos de proceso
        all_data = query.process_data()
        
        # Verificar que se devuelven exactamente 31 elementos
        self.assertEqual(len(all_data), 30, "El query no devolvió los 31 elementos esperados")
        
        # Verificar que el primer y último elemento contienen los datos esperados
        first = all_data[0]
        last = all_data[29]
        
        self.assertEqual(first.timestamp, "2023-10-01 08:00:00", "El timestamp del primer elemento no coincide")
        self.assertEqual(last.timestamp, "2023-10-03 17:00:00", "El timestamp del último elemento no coincide")
        
        # Verificar que todos los elementos tienen la estructura correcta
        for item in all_data:
            self.assertTrue(hasattr(item, "timestamp"), "Falta el atributo timestamp")
            self.assertTrue(hasattr(item, "temperature"), "Falta el atributo temperature")
            self.assertTrue(hasattr(item, "pressure"), "Falta el atributo pressure")
            self.assertTrue(hasattr(item, "velocity"), "Falta el atributo velocity")
            self.assertTrue(hasattr(item, "humidity"), "Falta el atributo humidity")
            self.assertTrue(hasattr(item, "qualityScore"), "Falta el atributo qualityScore")
    
    
    def test_mutation_create_optimization_results(self):
        """Prueba la Mutation para crear resultados de optimización."""
        # Guardar respaldo de los datos originales si existen
        original_data = None
        if os.path.exists(f"{self.env.temp_dir}/optimization_results.json"):
            with open(f"{self.env.temp_dir}/optimization_results.json", "r") as f:
                original_data = f.read()
        
        try:
            # Crear instancia de Mutation
            mutation = Mutation()
            
            # Ejecutar mutation para añadir resultados de optimización
            result = mutation.create_optimization_results(
                temperature=89.0,
                pressure=104.5,
                velocity=45.5,
                humidity=62.0,
                predictedQuality=91.2,
                confidenceScore=0.98
            )
            
            # Verificar que la mutation fue exitosa
            self.assertTrue(result, "La mutation no fue exitosa")
            
            # Verificar que se guardaron los datos correctamente
            optimization = load_optimization_results()
            self.assertIsNotNone(optimization, "No se cargaron los resultados de optimización")
            
            # Verificar los valores
            self.assertEqual(optimization.optimalParameters.temperature, 89.0, "La temperatura óptima no coincide")
            self.assertEqual(optimization.optimalParameters.pressure, 104.5, "La presión óptima no coincide")
            self.assertEqual(optimization.optimalParameters.velocity, 45.5, "La velocidad óptima no coincide")
            self.assertEqual(optimization.optimalParameters.humidity, 62.0, "La humedad óptima no coincide")
            self.assertEqual(optimization.predictedQuality, 91.2, "La calidad predicha no coincide")
            self.assertEqual(optimization.confidenceScore, 0.98, "El puntaje de confianza no coincide")
        
        finally:
            # Restaurar datos originales
            if original_data is not None:
                with open(f"{self.env.temp_dir}/optimization_results.json", "w") as f:
                    f.write(original_data)

if __name__ == "__main__":
    unittest.main() 