import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.services.optimize_service import run_optimization

class TestOptimizeService(unittest.TestCase):
    """Test cases for the optimization service"""

    def setUp(self):
        """Set up test data"""
        self.test_data = pd.DataFrame({
            'temperature': [85.3, 86.1, 87.2],
            'pressure': [102.4, 103.1, 102.8],
            'velocity': [45.1, 46.0, 45.5],
            'humidity': [65.2, 64.8, 64.5],
            'qualityScore': [87.3, 88.1, 89.0]
        })

    @patch('src.services.optimize_service.load_process_data')
    @patch('src.services.optimize_service.gp_minimize')
    def test_run_optimization(self, mock_gp_minimize, mock_load_data):
        """Test the main optimization function"""
        # Mock data loading
        mock_load_data.return_value = self.test_data
        
        # Mock optimization result
        mock_result = MagicMock()
        mock_result.x = np.array([90.0, 105.0, 45.0, 65.0])
        mock_result.fun = -95.0  # Negative because we minimize negative quality
        mock_gp_minimize.return_value = mock_result

        # Run optimization
        result = run_optimization()

        # Verify result structure and types
        self.assertIsInstance(result, dict)
        self.assertIn('best_params', result)
        self.assertIn('best_quality', result)
        self.assertIn('confidence_score', result)
        self.assertIn('correlation_matrix', result)
        self.assertIn('feature_importance', result)
        self.assertIn('model_performance', result)
        self.assertIn('recommendations', result)

        # Verify parameter types
        self.assertIsInstance(result['best_params'], list)
        self.assertEqual(len(result['best_params']), 4)  # 4 parameters
        self.assertIsInstance(result['best_quality'], float)
        self.assertIsInstance(result['confidence_score'], float)
        self.assertIsInstance(result['correlation_matrix'], dict)
        self.assertIsInstance(result['feature_importance'], list)
        self.assertIsInstance(result['model_performance'], float)
        self.assertIsInstance(result['recommendations'], list)

        # Verify value ranges
        for param in result['best_params']:
            self.assertIsInstance(param, float)
        self.assertGreaterEqual(result['confidence_score'], 0.0)
        self.assertLessEqual(result['confidence_score'], 1.0)
        self.assertGreaterEqual(result['model_performance'], 0.0)
        self.assertLessEqual(result['model_performance'], 1.0)

        # Verify feature importance
        self.assertEqual(len(result['feature_importance']), 4)  # 4 features
        self.assertAlmostEqual(sum(result['feature_importance']), 1.0, places=5)

        # Verify optimization was called
        mock_gp_minimize.assert_called_once()
        mock_load_data.assert_called_once()

    @patch('src.services.optimize_service.load_process_data')
    def test_run_optimization_with_empty_data(self, mock_load_data):
        """Test optimization with empty dataset"""
        # Mock empty data
        mock_load_data.return_value = pd.DataFrame()
        
        # Verify raises exception
        with self.assertRaises(ValueError):
            run_optimization()

if __name__ == '__main__':
    unittest.main()
