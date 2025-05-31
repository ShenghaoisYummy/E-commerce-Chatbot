#!/usr/bin/env python3
"""
Deployment script for E-commerce Chatbot
Compares current model with historical best and deploys if it's the best performing model
"""

import os
import sys
import json
import argparse
import logging
import mlflow
from datetime import datetime
from typing import Dict, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_selection import (
    compare_with_history, 
    should_deploy_model, 
    log_comparison_to_mlflow
)
from utils.mlflow_utils import mlflow_start_run, mlflow_setup_tracking
from utils.yaml_utils import load_config
from src.deployment import deploy_model_to_huggingface
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Model deployment script")
    parser.add_argument(
        "--evaluation-results", 
        type=str, 
        default="results/evaluations/metrics.json",
        help="Path to evaluation results JSON file"
    )
    parser.add_argument(
        "--model-artifact-path", 
        type=str, 
        default="results/fine_tuned_model_location.json",
        help="Path to model artifact location file"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="params.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="results/deployment",
        help="Directory to save deployment results"
    )
    parser.add_argument(
        "--min-improvement-threshold", 
        type=float, 
        default=0.001,
        help="Minimum improvement threshold for deployment"
    )
    parser.add_argument(
        "--hf-model-id", 
        type=str, 
        default="ShenghaoYummy/TinyLlama-ECommerce-Chatbot",
        help="Hugging Face model ID for deployment"
    )
    parser.add_argument(
        "--skip-hf-push", 
        action="store_true",
        help="Skip pushing to Hugging Face Hub"
    )
    parser.add_argument(
        "--force-deploy", 
        action="store_true",
        help="Force deployment regardless of performance comparison"
    )
    
    return parser.parse_args()

def load_evaluation_results(results_path: str) -> Dict[str, float]:
    """Load evaluation results from JSON file"""
    try:
        with open(results_path, 'r') as f:
            metrics = json.load(f)
        
        logger.info(f"Loaded evaluation results from {results_path}")
        logger.info(f"Available metrics: {list(metrics.keys())}")
        
        # Validate required metrics are present
        required_metrics = ['bleu1', 'bleu2', 'bleu3', 'bleu4', 'rougeL_precision']
        missing_metrics = [m for m in required_metrics if m not in metrics]
        
        if missing_metrics:
            logger.warning(f"Missing required metrics: {missing_metrics}")
        
        return metrics
        
    except FileNotFoundError:
        logger.error(f"Evaluation results file not found: {results_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing evaluation results JSON: {e}")
        raise

def register_model_in_mlflow(model_info_path: str, 
                           comparison_result: Dict,
                           model_name: str = "ecommerce-chatbot") -> Optional[str]:
    """Register the model in MLflow Model Registry"""
    try:
        # Load model information
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        
        run_id = model_info.get('mlflow_run_id')
        if not run_id:
            logger.error("MLflow run ID not found in model info")
            return None
        
        # Create model version description
        description = f"""
E-commerce Chatbot Model - Deployed {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Performance Metrics:
- Weighted Score: {comparison_result['current_score']:.4f}
- BLEU-4: {comparison_result['current_contributions'].get('bleu4', 0):.4f}
- ROUGE-L Precision: {comparison_result['current_contributions'].get('rougeL_precision', 0):.4f}

Improvement: {comparison_result['improvement']:.4f} ({comparison_result['improvement_percentage']:.2f}%)
"""
        
        # Register model
        client = mlflow.tracking.MlflowClient()
        
        # Create registered model if it doesn't exist
        try:
            client.create_registered_model(
                name=model_name,
                description="Fine-tuned TinyLlama model for e-commerce customer service"
            )
            logger.info(f"Created new registered model: {model_name}")
        except mlflow.exceptions.RestException:
            logger.info(f"Registered model {model_name} already exists")
        
        # Create model version
        model_version = client.create_model_version(
            name=model_name,
            source=f"runs:/{run_id}/model",
            description=description,
            tags={
                "deployment_timestamp": datetime.now().isoformat(),
                "weighted_score": str(comparison_result['current_score']),
                "is_best": str(comparison_result['is_best']),
                "improvement": str(comparison_result['improvement'])
            }
        )
        
        # Transition to Production stage
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production",
            archive_existing_versions=True
        )
        
        logger.info(f"✅ Model registered and promoted to Production: {model_name} v{model_version.version}")
        return model_version.version
        
    except Exception as e:
        logger.error(f"Error registering model in MLflow: {e}")
        return None

def deploy_to_huggingface(model_info_path: str, 
                         hf_model_id: str,
                         comparison_result: Dict) -> bool:
    """Deploy model to Hugging Face Hub"""
    try:
        logger.info(f"Starting deployment to Hugging Face Hub: {hf_model_id}")
        
        # Prepare performance metrics for model card
        performance_metrics = {
            'current_score': comparison_result['current_score'],
            'bleu1': comparison_result['current_contributions'].get('bleu1', 0),
            'bleu2': comparison_result['current_contributions'].get('bleu2', 0),
            'bleu3': comparison_result['current_contributions'].get('bleu3', 0),
            'bleu4': comparison_result['current_contributions'].get('bleu4', 0),
            'rougeL_precision': comparison_result['current_contributions'].get('rougeL_precision', 0)
        }
        
        # Use the deployment module function
        success = deploy_model_to_huggingface(
            model_info_path=model_info_path,
            hf_model_id=hf_model_id,
            commit_message=f"Deploy best model - Score: {comparison_result['current_score']:.4f}",
            performance_metrics=performance_metrics,
            merge_adapter=True
        )
        
        if success:
            logger.info(f"✅ Model successfully deployed to Hugging Face: https://huggingface.co/{hf_model_id}")
            return True
        else:
            logger.error("❌ Failed to deploy model to Hugging Face")
            return False
            
    except Exception as e:
        logger.error(f"Error deploying to Hugging Face: {e}")
        return False

def save_deployment_results(output_dir: str, 
                          comparison_result: Dict,
                          deployment_info: Dict):
    """Save deployment results to output directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save comparison results
    comparison_file = os.path.join(output_dir, "model_comparison.json")
    with open(comparison_file, 'w') as f:
        json.dump(comparison_result, f, indent=2)
    
    # Save deployment info
    deployment_file = os.path.join(output_dir, "deployment_info.json")
    with open(deployment_file, 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    logger.info(f"Deployment results saved to {output_dir}")

def main():
    """Main deployment function"""
    args = parse_args()
    
    logger.info("🚀 Starting model deployment process...")
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up MLflow tracking
    mlflow_setup_tracking(config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start MLflow run for deployment
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"deployment_{run_timestamp}"
    
    with mlflow_start_run(run_name) as run:
        deployment_run_id = run.info.run_id
        logger.info(f"Started deployment run: {deployment_run_id}")
        
        try:
            # Step 1: Load evaluation results
            logger.info("📊 Step 1: Loading evaluation results...")
            evaluation_metrics = load_evaluation_results(args.evaluation_results)
            
            # Step 2: Compare with historical best
            logger.info("📈 Step 2: Comparing with historical performance...")
            experiment_name = config.get('mlflow', {}).get('experiment_name', 'ecommerce-chatbot-evaluation')
            comparison_result = compare_with_history(
                current_metrics=evaluation_metrics,
                experiment_name=experiment_name
            )
            
            # Log comparison results to MLflow
            log_comparison_to_mlflow(comparison_result, deployment_run_id)
            
            # Step 3: Decide whether to deploy
            logger.info("🤔 Step 3: Making deployment decision...")
            should_deploy = args.force_deploy or should_deploy_model(
                comparison_result, 
                args.min_improvement_threshold
            )
            
            deployment_info = {
                "deployment_timestamp": datetime.now().isoformat(),
                "deployment_run_id": deployment_run_id,
                "should_deploy": should_deploy,
                "force_deploy": args.force_deploy,
                "min_improvement_threshold": args.min_improvement_threshold,
                "mlflow_registered": False,
                "huggingface_deployed": False,
                "mlflow_model_version": None
            }
            
            if should_deploy:
                logger.info("✅ Model meets deployment criteria - proceeding with deployment")
                
                # Step 4: Register model in MLflow
                logger.info("📝 Step 4: Registering model in MLflow...")
                model_version = register_model_in_mlflow(
                    args.model_artifact_path, 
                    comparison_result
                )
                
                if model_version:
                    deployment_info["mlflow_registered"] = True
                    deployment_info["mlflow_model_version"] = model_version
                    mlflow.log_param("mlflow_model_version", model_version)
                
                # Step 5: Deploy to Hugging Face (if not skipped)
                if not args.skip_hf_push:
                    logger.info("🤗 Step 5: Deploying to Hugging Face Hub...")
                    hf_success = deploy_to_huggingface(
                        args.model_artifact_path,
                        args.hf_model_id,
                        comparison_result
                    )
                    deployment_info["huggingface_deployed"] = hf_success
                    mlflow.log_param("huggingface_deployed", hf_success)
                    mlflow.log_param("hf_model_id", args.hf_model_id)
                else:
                    logger.info("⏭️  Skipping Hugging Face deployment (--skip-hf-push)")
                
                logger.info("🎉 Deployment completed successfully!")
                
            else:
                logger.info("⏸️  Model does not meet deployment criteria - skipping deployment")
            
            # Step 6: Save results
            logger.info("💾 Step 6: Saving deployment results...")
            save_deployment_results(args.output_dir, comparison_result, deployment_info)
            
            # Log final deployment status
            mlflow.log_param("deployment_decision", "deploy" if should_deploy else "skip")
            mlflow.log_metric("deployment_success", 1.0 if should_deploy else 0.0)
            
            # Log deployment info as artifact
            deployment_artifact_path = os.path.join(args.output_dir, "deployment_info.json")
            mlflow.log_artifact(deployment_artifact_path)
            
            logger.info("✅ Deployment process completed successfully!")
            
            # Print summary
            print("\n" + "="*60)
            print("DEPLOYMENT SUMMARY")
            print("="*60)
            print(f"Current Score: {comparison_result['current_score']:.4f}")
            print(f"Historical Best: {comparison_result['historical_best_score']}")
            print(f"Is Best Model: {comparison_result['is_best']}")
            print(f"Should Deploy: {should_deploy}")
            if deployment_info["mlflow_registered"]:
                print(f"MLflow Model Version: {deployment_info['mlflow_model_version']}")
            if deployment_info["huggingface_deployed"]:
                print(f"Hugging Face Model: https://huggingface.co/{args.hf_model_id}")
            print("="*60)
            
        except Exception as e:
            logger.error(f"❌ Deployment process failed: {e}")
            mlflow.log_param("deployment_error", str(e))
            raise

if __name__ == "__main__":
    main() 