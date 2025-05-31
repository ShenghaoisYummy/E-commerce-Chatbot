import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import json
import logging
import mlflow
from datetime import datetime

logger = logging.getLogger(__name__)

def calculate_weighted_score(metrics: Dict[str, float], 
                           weights: Optional[Dict[str, float]] = None) -> Tuple[float, Dict[str, float]]:
    """
    Calculate weighted score using BLEU 1-4 and ROUGE-L precision
    
    Args:
        metrics: Dictionary of metric values
        weights: Optional custom weights, uses default if None
        
    Returns:
        Tuple of (weighted_score, contribution_breakdown)
    """
    # Default weights optimized for e-commerce chatbot
    if weights is None:
        weights = {
            'bleu1': 0.10,      # 10% - Basic vocabulary coverage
            'bleu2': 0.15,      # 15% - Phrase fluency  
            'bleu3': 0.25,      # 25% - Local grammar structure
            'bleu4': 0.40,      # 40% - Overall quality (most important)
            'rougeL_precision': 0.10  # 10% - Long sequence precision
        }
    
    # Validate weights sum to 1.0
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 1e-6:
        # Auto-normalize weights
        weights = {k: v/total_weight for k, v in weights.items()}
        logger.warning(f"Weights normalized to sum to 1.0")
    
    # Calculate weighted score
    total_score = 0.0
    contributions = {}
    
    for metric_name, weight in weights.items():
        metric_value = metrics.get(metric_name, 0.0)
        contribution = metric_value * weight
        total_score += contribution
        contributions[metric_name] = contribution
        
        logger.debug(f"{metric_name}: {metric_value:.4f} * {weight:.2f} = {contribution:.4f}")
    
    logger.info(f"Calculated weighted score: {total_score:.4f}")
    return total_score, contributions

def get_historical_best_score(experiment_name: str = "ecommerce-chatbot-evaluation") -> Tuple[Optional[float], Optional[str]]:
    """
    Get the best historical weighted score from MLflow
    
    Args:
        experiment_name: MLflow experiment name
        
    Returns:
        Tuple of (best_score, best_run_id) or (None, None) if no history
    """
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Get experiment
        try:
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                logger.info(f"No experiment found with name: {experiment_name}")
                return None, None
        except Exception as e:
            logger.info(f"Experiment not found: {e}")
            return None, None
        
        # Search for runs with weighted_score metric
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="metrics.weighted_score > 0",
            order_by=["metrics.weighted_score DESC"],
            max_results=1
        )
        
        if not runs:
            logger.info("No previous runs with weighted_score found")
            return None, None
        
        best_run = runs[0]
        best_score = best_run.data.metrics.get('weighted_score')
        best_run_id = best_run.info.run_id
        
        logger.info(f"Historical best score: {best_score:.4f} from run {best_run_id}")
        return best_score, best_run_id
        
    except Exception as e:
        logger.error(f"Error retrieving historical best score: {e}")
        return None, None

def compare_with_history(current_metrics: Dict[str, float], 
                        experiment_name: str = "ecommerce-chatbot-evaluation",
                        weights: Optional[Dict[str, float]] = None) -> Dict[str, any]:
    """
    Compare current model performance with historical best
    
    Args:
        current_metrics: Current model's evaluation metrics
        experiment_name: MLflow experiment name
        weights: Optional custom weights for scoring
        
    Returns:
        Dictionary with comparison results
    """
    # Calculate current weighted score
    current_score, contributions = calculate_weighted_score(current_metrics, weights)
    
    # Get historical best
    historical_best_score, best_run_id = get_historical_best_score(experiment_name)
    
    # Determine if current model is best
    is_best = historical_best_score is None or current_score > historical_best_score
    
    improvement = 0.0
    if historical_best_score is not None:
        improvement = current_score - historical_best_score
    
    comparison_result = {
        'current_score': current_score,
        'current_contributions': contributions,
        'historical_best_score': historical_best_score,
        'best_run_id': best_run_id,
        'is_best': is_best,
        'improvement': improvement,
        'improvement_percentage': (improvement / historical_best_score * 100) if historical_best_score else 0.0,
        'timestamp': datetime.now().isoformat()
    }
    
    if is_best:
        if historical_best_score is None:
            logger.info("🎉 This is the first model - automatically considered best!")
        else:
            logger.info(f"🎉 New best model! Improvement: +{improvement:.4f} ({comparison_result['improvement_percentage']:.2f}%)")
    else:
        logger.info(f"❌ Current model ({current_score:.4f}) did not beat historical best ({historical_best_score:.4f})")
    
    return comparison_result

def log_comparison_to_mlflow(comparison_result: Dict[str, any], run_id: Optional[str] = None):
    """
    Log comparison results to MLflow
    
    Args:
        comparison_result: Results from compare_with_history
        run_id: Optional specific run ID to log to
    """
    try:
        if run_id:
            with mlflow.start_run(run_id=run_id):
                _log_comparison_metrics(comparison_result)
        else:
            _log_comparison_metrics(comparison_result)
            
    except Exception as e:
        logger.error(f"Error logging comparison to MLflow: {e}")

def _log_comparison_metrics(comparison_result: Dict[str, any]):
    """Helper function to log comparison metrics"""
    # Log main metrics
    mlflow.log_metric("weighted_score", comparison_result['current_score'])
    mlflow.log_metric("is_best_model", 1.0 if comparison_result['is_best'] else 0.0)
    
    if comparison_result['historical_best_score'] is not None:
        mlflow.log_metric("historical_best_score", comparison_result['historical_best_score'])
        mlflow.log_metric("improvement", comparison_result['improvement'])
        mlflow.log_metric("improvement_percentage", comparison_result['improvement_percentage'])
    
    # Log contribution breakdown
    for metric, contribution in comparison_result['current_contributions'].items():
        mlflow.log_metric(f"contribution_{metric}", contribution)
    
    # Log comparison as artifact
    comparison_file = "model_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison_result, f, indent=2)
    mlflow.log_artifact(comparison_file)

def should_deploy_model(comparison_result: Dict[str, any], 
                       min_improvement_threshold: float = 0.001) -> bool:
    """
    Determine if model should be deployed based on comparison results
    
    Args:
        comparison_result: Results from compare_with_history
        min_improvement_threshold: Minimum improvement required for deployment
        
    Returns:
        Boolean indicating if model should be deployed
    """
    if comparison_result['is_best']:
        if comparison_result['historical_best_score'] is None:
            # First model - deploy
            logger.info("✅ Deploying: First model in history")
            return True
        elif comparison_result['improvement'] >= min_improvement_threshold:
            # Significant improvement - deploy
            logger.info(f"✅ Deploying: Improvement {comparison_result['improvement']:.4f} >= threshold {min_improvement_threshold}")
            return True
        else:
            # Improvement too small - don't deploy
            logger.info(f"❌ Not deploying: Improvement {comparison_result['improvement']:.4f} < threshold {min_improvement_threshold}")
            return False
    else:
        # Not the best model - don't deploy
        logger.info("❌ Not deploying: Not the best model")
        return False

# Example usage function
def example_model_comparison():
    """Example of how to use the model comparison functionality"""
    
    # Simulate current model metrics
    current_metrics = {
        'bleu1': 0.3713,
        'bleu2': 0.2622,
        'bleu3': 0.2035,
        'bleu4': 0.1603,
        'rougeL_precision': 0.6641
    }
    
    # Compare with history
    comparison_result = compare_with_history(current_metrics)
    
    # Check if should deploy
    should_deploy = should_deploy_model(comparison_result)
    
    print(f"Should deploy: {should_deploy}")
    print(f"Comparison result: {json.dumps(comparison_result, indent=2)}")
    
    return comparison_result, should_deploy

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run example
    example_model_comparison() 