import pandas as pd
import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from skopt import gp_minimize

# Constants
RANDOM_FOREST_REGRESSOR = 42
RESULTS_DIR = 'results'

def run_optimization():
    """
    Main optimization method that orchestrates the optimization process
    
    Returns:
        dict: Dictionary containing best parameters and predicted quality
    """
    data = load_process_data()
    corr = descriptive_analysis(data)
    
    # Define predictor variables and target variable
    features = ['temperature', 'pressure', 'velocity', 'humidity']
    target = 'qualityScore'
    X = data[features]
    y = data[target]
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_FOREST_REGRESSOR)
    
    # Scikit-learn pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(random_state=RANDOM_FOREST_REGRESSOR))
    ])
    
    # Optimization with GridSearchCV to tune hyperparameters
    param_grid = {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [None, 10, 20],
        'regressor__min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)
    
    logging.info("\nBest hyperparameters found (GridSearchCV):")
    logging.info(grid_search.best_params_)
    logging.info("Best training score (GridSearchCV): %s", grid_search.best_score_)
    
    # Model validation
    y_pred = grid_search.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    logging.info("\nValidation on test set:")
    logging.info("R²: %s", r2)
    logging.info("Mean Squared Error: %s", mse)
    
    # Get feature importance from the best model
    best_model = grid_search.best_estimator_.named_steps['regressor']
    feature_importance = best_model.feature_importances_
    
    logging.info("\nFeature Importance:")
    for feature, importance in zip(features, feature_importance):
        logging.info("%s: %s", feature, importance)
    
    # Search space for each variable
    space = [
        (80, 100),    # temperature
        (100, 110),   # pressure
        (40, 50),     # velocity
        (60, 70)      # humidity
    ]
    
    # Run Bayesian optimization with gp_minimize
    res = gp_minimize(
        func=lambda x: objective(x, features, grid_search), 
        dimensions=space, 
        n_calls=50, 
        random_state=RANDOM_FOREST_REGRESSOR
    )

    best_params = [float(param) for param in res.x]  # Convert numpy values to Python floats
    best_quality = float(-res.fun)  # Convert minimization to maximum quality value and to float
    
    logging.info("\nBest optimized parameters (gp_minimize):")
    logging.info("Temperature: %s, Pressure: %s, Velocity: %s, Humidity: %s", 
                best_params[0], best_params[1], best_params[2], best_params[3])
    logging.info("Predicted optimal quality: %s", best_quality)

    # Calculate model performance as average of R2 and grid search best score
    model_performance = float((r2 + grid_search.best_score_) / 2)  # Convert to float

    recommendations = get_recommendations(features, feature_importance, best_params, X.mean())
    
    logging.info("\nRecommendations:")
    for rec in recommendations:
        logging.info("- %s", rec)

    return {
        'best_params': best_params,
        'best_quality': best_quality,
        'confidence_score': float(r2),  # Convert to float
        'correlation_matrix': corr.to_dict(),  # Convert DataFrame to dict
        'feature_importance': [float(imp) for imp in feature_importance],  # Convert numpy values to Python floats
        'model_performance': model_performance,
        'recommendations': recommendations
    }

def load_process_data() -> pd.DataFrame:
    """
    Load process data from JSON file.
    
    Returns:
        pandas.DataFrame: DataFrame containing process data
    """
    return pd.read_json(f"{RESULTS_DIR}/process_data.json", orient='records')

def descriptive_analysis(data: pd.DataFrame) -> pd.DataFrame:
    """
    Performs descriptive analysis and correlation matrix calculation on process data.
    Prints descriptive statistics and correlation matrix, and returns the correlation matrix.
    
    Args:
        data (pd.DataFrame): DataFrame containing process variables
        
    Returns:
        pd.DataFrame: Correlation matrix of the process variables
    """
    corr = data.corr()
    logging.info("[descriptive_analysis]: Data Description:")
    logging.info(data.describe())
    
    logging.info("[descriptive_analysis]: Correlation Matrix:")
    logging.info(corr)

    return corr

def objective(params: list, features: list, grid_search: object) -> float:
    """
    Objective function for optimization that predicts quality score for given parameters.

    Args:
        params: List of process parameters [temperature, pressure, velocity, humidity]
        features: List of feature names
        grid_search: Trained grid search model object

    Returns:
        float: Negative predicted quality score (for minimization)
    """
    temperature, pressure, velocity, humidity = params
    new_x = pd.DataFrame([[temperature, pressure, velocity, humidity]], columns=features)
    quality_pred = grid_search.predict(new_x)[0]
    return -quality_pred  # Minimize to maximize quality


def get_recommendations(
    features: list, 
    feature_importance: list,
    optimal_params: list, 
    current_means: list
) -> list:
    """
    Generate recommendations by comparing current parameter values with optimal ones.
    
    Analyzes the difference between current and optimal parameter values, taking into
    account feature importance, to generate actionable recommendations for process
    optimization.

    Args:
        features: List of feature names
        feature_importance: List of importance scores for each feature
        optimal_params: List of optimal parameter values found by optimization
        current_means: List of current mean values for each parameter

    Returns:
        list: List of recommendation strings for process optimization
    """
    recommendations = []
    
    for feature, importance, optimal, current in zip(features, feature_importance, optimal_params, current_means):
        if importance > 0.05:  # Reduced threshold to catch more features
            diff = optimal - current
            if abs(diff) > 0.05 * current:  # Reduced threshold for recommendations
                direction = "increase" if diff > 0 else "decrease"
                percent_change = abs(diff/current * 100)
                recommendations.append(
                    f"Recommend to {direction} {feature} by approximately "
                    f"{percent_change:.1f}% (from {current:.1f} to {optimal:.1f})"
                )

    if not recommendations:
        recommendations.append("Current parameters are within optimal ranges")
    return recommendations