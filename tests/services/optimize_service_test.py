import pytest
import pandas as pd
import numpy as np
from src.services.optimize_service import (
    run_optimization,
    load_process_data,
    descriptive_analysis,
    objective,
    get_recommendations
)

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'timestamp': ['2023-10-01T08:00:00', '2023-10-01T09:00:00'],
        'temperature': [85.3, 86.1],
        'pressure': [102.4, 103.1], 
        'velocity': [45.1, 46.0],
        'humidity': [65.2, 64.8],
        'qualityScore': [87.3, 88.1]
    })

def test_load_process_data():
    data = load_process_data()
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert all(col in data.columns for col in ['temperature', 'pressure', 'velocity', 'humidity', 'qualityScore'])

def test_descriptive_analysis(sample_data):
    corr = descriptive_analysis(sample_data)
    assert isinstance(corr, pd.DataFrame)
    assert corr.shape == (5, 5)  # 5x5 correlation matrix for numeric variables
    assert all(col in corr.columns for col in ['temperature', 'pressure', 'velocity', 'humidity', 'qualityScore'])

def test_objective():
    # Mock grid search object
    class MockGridSearch:
        def predict(self, X):
            return np.array([90.0])
    
    mock_grid_search = MockGridSearch()
    features = ['temperature', 'pressure', 'velocity', 'humidity']
    params = [85.0, 102.0, 45.0, 65.0]
    
    result = objective(params, features, mock_grid_search)
    assert isinstance(result, float)
    assert result == -90.0  # Negative because we minimize for optimization

def test_get_recommendations():
    features = ['temperature', 'pressure', 'velocity', 'humidity']
    feature_importance = [0.3, 0.2, 0.1, 0.4]
    optimal_params = [90.0, 105.0, 47.0, 63.0]
    current_means = [85.0, 102.0, 45.0, 65.0]
    
    recommendations = get_recommendations(features, feature_importance, optimal_params, current_means)
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    assert all(isinstance(rec, str) for rec in recommendations)

def test_run_optimization():
    result = run_optimization()
    assert isinstance(result, dict)
    expected_keys = [
        'best_params',
        'best_quality',
        'confidence_score',
        'correlation_matrix',
        'feature_importance',
        'model_performance',
        'recommendations'
    ]
    assert all(key in result for key in expected_keys)
    assert len(result['best_params']) == 4
    assert isinstance(result['best_quality'], float)
    assert isinstance(result['confidence_score'], float)
    assert isinstance(result['correlation_matrix'], dict)
    assert len(result['feature_importance']) == 4
    assert isinstance(result['model_performance'], float)
    assert isinstance(result['recommendations'], list)

def test_run_optimization_parameter_ranges():
    result = run_optimization()
    
    # Test temperature range (80-100)
    assert 80 <= result['best_params'][0] <= 100
    
    # Test pressure range (100-110)
    assert 100 <= result['best_params'][1] <= 110
    
    # Test velocity range (40-50)
    assert 40 <= result['best_params'][2] <= 50
    
    # Test humidity range (60-70)
    assert 60 <= result['best_params'][3] <= 70
    
    # Test quality score range (typically 0-100)
    assert 0 <= result['best_quality'] <= 100
    
    # Test confidence and performance scores (0-1 range)
    assert 0 <= result['confidence_score'] <= 1
    assert 0 <= result['model_performance'] <= 1
