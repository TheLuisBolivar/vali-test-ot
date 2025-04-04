# Documentación API GraphQL - Sistema de Optimización de Proceso Industrial

## Introducción

Esta API GraphQL permite consultar y actualizar datos relacionados con el sistema de optimización de proceso industrial. Proporciona acceso a:

- Datos históricos del proceso (temperatura, presión, velocidad, humedad, calidad)
- Resultados del análisis de datos y correlaciones
- Parámetros óptimos recomendados por el algoritmo
- Funcionalidad para publicar nuevos resultados del algoritmo

## Endpoint
com
La API está disponible en:

```
http://localhost:8000/graphql
```

## Tipos de Datos

### ProcessVariable

Representa los datos de una medición del proceso en un momento determinado.

```graphql
type ProcessVariable {
  timestamp: String!
  temperature: Float!
  pressure: Float!
  velocity: Float!
  humidity: Float!
  qualityScore: Float!
}
```

### OptimalParameters

Representa los parámetros óptimos recomendados por el algoritmo.

```graphql
type OptimalParameters {
  temperature: Float!
  pressure: Float!
  velocity: Float!
  humidity: Float!
}
```

### OptimizationResult

Resultado completo de la optimización de parámetros.

```graphql
type OptimizationResult {
  optimalParameters: OptimalParameters!
  predictedQuality: Float!
  confidenceScore: Float!
  optimizationTime: Float!
}
```

### AnalysisResult

Resultado del análisis de datos del proceso.

```graphql
type AnalysisResult {
  correlationMatrix: [[Float!]!]!
  featureImportance: [Float!]!
  modelPerformance: Float!
  recommendations: [String!]!
}
```

