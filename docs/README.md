# Prueba Técnica - Sr. Software Developer

## Contexto
En una planta de manufactura, se utiliza un sistema de control que requiere mantener ciertas variables críticas en niveles óptimos para garantizar la calidad del producto final. El sistema actual utiliza un algoritmo de optimización que sugiere los parámetros óptimos para estas variables.

Recientemente, el algoritmo ha comenzado a mostrar comportamientos anómalos y sus sugerencias ya no están optimizando correctamente el proceso.

## Objetivo
Desarrollar una solución que permita:
1. Analizar el comportamiento actual del algoritmo
2. Identificar posibles causas del deterioro en las predicciones
3. Proponer y justificar una solución técnica
4. Implementar una solución que mejore el rendimiento del sistema
5. Consumir y producir datos del servidor GraphQL proporcionado

## Requisitos Técnicos
- Python 3.x
- Al menos una de las siguientes bibliotecas: scikit-learn, TensorFlow, PyTorch, o scipy
- Tests unitarios y de integración
- Consumo de datos desde el servidor GraphQL proporcionado

## Datos
Se proporciona un servidor GraphQL que expone:
- Variables de entrada (temperatura, presión, velocidad, humedad)
- Variable de salida (calidad del producto)
- Timestamps de las mediciones

El servidor GraphQL está disponible en `http://localhost:8000/graphql`

## Esquema GraphQL Proporcionado
El servidor GraphQL expone los siguientes tipos y consultas:

```graphql
# Tipos
type ProcessVariable {
  timestamp: String!
  temperature: Float!
  pressure: Float!
  velocity: Float!
  humidity: Float!
  qualityScore: Float!
}

# Consultas
query {
  processData {
    timestamp
    temperature
    pressure
    velocity
    humidity
    qualityScore
  }
}
```

## Entregables
1. Código fuente completo y documentado que incluya:
   - Análisis de datos y correlaciones
   - Modelo de machine learning
   - Optimización de parámetros
   - Sistema de recomendaciones
   - Cliente GraphQL para consumir y producir datos al servidor proporcionado
2. Documentación técnica que incluya:
   - Análisis del problema
   - Justificación de la solución propuesta
   - Métricas de rendimiento
   - Instrucciones de despliegue
3. Tests unitarios y de integración

## Tiempo
La prueba debe completarse en un día (8 horas).

## Iniciar el Servidor GraphQL
Para iniciar el servidor GraphQL que proporciona los datos, ejecute:

```
poetry run python src/graphql_server.py
```

## Criterios de Evaluación
- Calidad del código y arquitectura
- Uso de buenas prácticas de programación
- Documentación clara y completa
- Cobertura de pruebas
- Solución técnica propuesta
- Capacidad de análisis y resolución de problemas

## Contacto
Si tienes alguna duda sobre la prueba técnica, por favor contacta a:
- Email: [ricardodavila@valiot.io] 