"""
# GraphQL Server para Sistema de Optimización de Proceso Industrial

Este servidor GraphQL expone:
1. Datos históricos del proceso (temperatura, presión, velocidad, humedad, calidad)
2. Resultados del análisis de datos y correlaciones
3. Parámetros óptimos recomendados por el algoritmo
4. Mutations para publicar nuevos resultados del algoritmo

## Uso
- Iniciar el servidor: python src/graphql_server.py
- Acceder al GraphQL Playground: http://localhost:8000/graphql
"""

import strawberry
import json
import os
from typing import List, Optional, Dict, Any, Union
from strawberry.asgi import GraphQL
from datetime import datetime
import uvicorn
from fastapi import FastAPI
from services.optimize_service import run_optimization
import time

# Directorio donde se almacenan los resultados
RESULTS_DIR = 'results'

# Definición de tipos para el esquema GraphQL
@strawberry.type
class ProcessVariable:
    timestamp: str
    temperature: float
    pressure: float
    velocity: float
    humidity: float
    qualityScore: float

@strawberry.type
class OptimalParameters:
    temperature: float
    pressure: float
    velocity: float
    humidity: float

@strawberry.type
class OptimizationResult:
    optimalParameters: OptimalParameters
    predictedQuality: float
    confidenceScore: float
    optimizationTime: float

@strawberry.type
class AnalysisResult:
    correlationMatrix: List[List[float]]
    featureImportance: List[float]
    modelPerformance: float
    recommendations: List[str]

@strawberry.input
class ProcessVariableInput:
    temperature: float
    pressure: float
    velocity: float
    humidity: float
    qualityScore: Optional[float] = None

# Funciones de utilidad para cargar datos
def load_process_data():
    """Carga datos históricos del proceso desde JSON."""
    try:
        if os.path.exists(f'{RESULTS_DIR}/process_data.json'):
            data = pd.read_json(f'{RESULTS_DIR}/process_data.json', orient='records')
            return [
                ProcessVariable(
                    timestamp=str(row['timestamp']),
                    temperature=row['temperature'],
                    pressure=row['pressure'],
                    velocity=row['velocity'],
                    humidity=row['humidity'],
                    qualityScore=row['qualityScore']
                )
                for _, row in data.iterrows()
            ]
        return []
    except Exception as e:
        print(f"Error loading process data: {e}")
        return []

def load_optimization_results():
    """Carga resultados de optimización desde JSON."""
    try:
        if os.path.exists(f'{RESULTS_DIR}/optimization_results.json'):
            with open(f'{RESULTS_DIR}/optimization_results.json', 'r') as f:
                data = json.load(f)
                
            return OptimizationResult(
                optimalParameters=OptimalParameters(
                    temperature=data['optimalParameters']['temperature'],
                    pressure=data['optimalParameters']['pressure'],
                    velocity=data['optimalParameters']['velocity'],
                    humidity=data['optimalParameters']['humidity']
                ),
                predictedQuality=data['predictedQuality'],
                confidenceScore=data['confidenceScore'],
                optimizationTime=data['optimizationTime']
            )
        return None
    except Exception as e:
        print(f"Error loading optimization results: {e}")
        return None

def load_analysis_results():
    """Carga resultados de análisis desde JSON."""
    try:
        if os.path.exists(f'{RESULTS_DIR}/analysis_results.json'):
            with open(f'{RESULTS_DIR}/analysis_results.json', 'r') as f:
                data = json.load(f)
                
            return AnalysisResult(
                correlationMatrix=data['correlationMatrix'],
                featureImportance=data['featureImportance'],
                modelPerformance=data['modelPerformance'],
                recommendations=data['recommendations']
            )
        return None
    except Exception as e:
        print(f"Error loading analysis results: {e}")
        return None

def save_optimization_results(results):
    """Guarda resultados de optimización en JSON."""
    try:
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
            
        with open(f'{RESULTS_DIR}/optimization_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving optimization results: {e}")
        return False

def save_analysis_results(results):
    """Guarda resultados de análisis en JSON."""
    try:
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
            
        with open(f'{RESULTS_DIR}/analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving analysis results: {e}")
        return False

def append_process_data(data):
    """Añade nuevos datos de proceso al historial."""
    try:
        # Crear archivo si no existe
        if not os.path.exists(f'{RESULTS_DIR}/process_data.json'):
            if not os.path.exists(RESULTS_DIR):
                os.makedirs(RESULTS_DIR)
            pd.DataFrame([]).to_json(f'{RESULTS_DIR}/process_data.json', orient='records')
        
        # Cargar datos existentes
        existing_data = pd.read_json(f'{RESULTS_DIR}/process_data.json', orient='records')
        
        # Crear nuevo registro
        new_data = pd.DataFrame([{
            'timestamp': datetime.now().isoformat(),
            'temperature': data.temperature,
            'pressure': data.pressure,
            'velocity': data.velocity,
            'humidity': data.humidity,
            'qualityScore': data.qualityScore or 0.0
        }])
        
        # Combinar y guardar
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
        combined_data.to_json(f'{RESULTS_DIR}/process_data.json', orient='records', date_format='iso')
        
        return True
    except Exception as e:
        print(f"Error appending process data: {e}")
        return False

# Definición del esquema GraphQL
@strawberry.type
class Query:
    @strawberry.field
    def process_data(self) -> List[ProcessVariable]:
        """Obtiene todos los datos históricos del proceso."""
        return load_process_data()
    
    @strawberry.field
    def latest_process_data(self, count: int = 10) -> List[ProcessVariable]:
        """Obtiene los últimos N registros de datos del proceso."""
        data = load_process_data()
        return data[-count:] if data else []
    
    @strawberry.field
    def optimization_results(self) -> Optional[OptimizationResult]:
        """Obtiene los resultados de la optimización de parámetros."""
        return load_optimization_results()
    
    @strawberry.field
    def analysis_results(self) -> Optional[AnalysisResult]:
        """Obtiene los resultados del análisis de datos."""
        return load_analysis_results()

@strawberry.type
class Mutation:
    @strawberry.mutation
    def create_optimization_results(
        self, 
        temperature: float, 
        pressure: float, 
        velocity: float, 
        humidity: float,
        predicted_quality: float,
        confidence_score: float,
        optimization_time: float,
    ) -> bool:
        """Actualiza los resultados de optimización con nuevos parámetros."""
        results = {
            'optimalParameters': {
                'temperature': temperature,
                'pressure': pressure,
                'velocity': velocity,
                'humidity': humidity
            },
            'predictedQuality': predicted_quality,
            'confidenceScore': confidence_score,
            'optimizationTime': optimization_time
        }
        return save_optimization_results(results)
    
    @strawberry.mutation
    def add_process_data(self, data: ProcessVariableInput) -> bool:
        """Añade un nuevo registro de datos de proceso."""
        return append_process_data(data)
    
    @strawberry.mutation
    def create_analysis_results(
        self,
        correlationMatrix: List[List[float]],
        featureImportance: List[float],
        modelPerformance: float,
        recommendations: List[str]
    ) -> bool:
        """Actualiza los resultados de análisis con nuevos datos."""
        results = {
            'correlationMatrix': correlationMatrix,
            'featureImportance': featureImportance,
            'modelPerformance': modelPerformance,
            'recommendations': recommendations
        }
        return save_analysis_results(results)

    @strawberry.mutation
    def run_optimization(self) -> bool:
        start_time = time.time()
        optimization_results = run_optimization()
        optimization_time = time.time() - start_time
        best_params = optimization_results['best_params']
        best_quality = optimization_results['best_quality']
        confidence_score = optimization_results['confidence_score']        
        correlation_matrix = optimization_results['confidence_score']
        feature_importance = optimization_results['feature_importance']
        model_performance = optimization_results['model_performance']
        recomendations = optimization_results['recommendations']
        
        # Usar las funciones de utilidad directamente
        save_optimization_results({
            'optimalParameters': {
                'temperature': best_params[0],
                'pressure': best_params[1],
                'velocity': best_params[2],
                'humidity': best_params[3]
            },
            'predictedQuality': best_quality,
            'confidenceScore': confidence_score,
            'optimizationTime': optimization_time
        })
        
        save_analysis_results({
            'correlationMatrix': correlation_matrix,
            'featureImportance': feature_importance,
            'modelPerformance': model_performance,
            'recommendations': recomendations
        })
        
        return True

# Crear esquema GraphQL
schema = strawberry.Schema(query=Query, mutation=Mutation)

# Crear aplicación FastAPI con GraphQL
app = FastAPI()
graphql_app = GraphQL(schema)
app.add_route("/graphql", graphql_app)
app.add_websocket_route("/graphql", graphql_app)

# Documentación de la API
@app.get("/")
def read_root():
    return {
        "title": "Sistema de Optimización de Proceso Industrial - API GraphQL",
        "description": "API para consultar y actualizar datos del proceso industrial",
        "graphql_endpoint": "/graphql"
    }

def main():
    """Inicia el servidor GraphQL."""
    print("Iniciando servidor GraphQL en http://localhost:8000/graphql")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main() 
