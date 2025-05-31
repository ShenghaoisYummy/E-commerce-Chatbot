#!/usr/bin/env python3
# scripts/evaluation_script.py
import sys
import os
import argparse
import mlflow
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import run_evaluation_pipeline
from utils.mlflow_utils import mlflow_start_run
from utils.yaml_utils import load_config
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument("--eval-dataset-path", type=str, default="data/processed/chatml_dataset_fine_tuning_dataset.jsonl", help="Path to evaluation dataset (ChatML JSONL format)")
    parser.add_argument("--model-artifact-path", type=str, default="results/fine_tuned_model_location.json", help="Path to model artifact file")
    parser.add_argument("--output-dir", type=str, default="results/evaluations", help="Directory to write evaluation results")
    parser.add_argument("--config", type=str, default="params.yaml", help="Path to YAML config file defining evaluation options")
    parser.add_argument("--mlflow-uri", type=str, default="", help="MLflow Tracking Server URI")
    parser.add_argument("--eval-size", type=int, default=100, help="Number of samples to evaluate (for faster testing)")
    parser.add_argument("--use-base-model", action="store_true", help="Use base model only (skip fine-tuned model)")

    return parser.parse_args()

def main(args):
    # Load configuration
    config = load_config()
    
    # Generate a run ID and set up directories
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"chatbot_evaluation_{run_timestamp}"
    
    # Start MLflow run
    with mlflow_start_run(run_name) as run:
        run_id = run.info.run_id
        print(f"MLflow run ID: {run_id}")
        
        # Log parameters
        mlflow.log_param("eval_dataset_path", args.eval_dataset_path)
        mlflow.log_param("model_artifact_path", args.model_artifact_path)
        mlflow.log_param("eval_size", args.eval_size)
        mlflow.log_param("use_base_model", args.use_base_model)
        
        # Run evaluation pipeline
        try:
            metrics, results_df = run_evaluation_pipeline(
                eval_dataset_path=args.eval_dataset_path,
                model_artifact_path=args.model_artifact_path,
                output_dir=args.output_dir,
                config=config,
                eval_size=args.eval_size,
                use_base_model=args.use_base_model
            )
            
            # Log metrics to MLflow
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
                
            # Log artifacts
            metrics_file = os.path.join(args.output_dir, "metrics.json")
            results_file = os.path.join(args.output_dir, "evaluation_results.csv")
            mlflow.log_artifact(metrics_file)
            mlflow.log_artifact(results_file)
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            mlflow.log_param("evaluation_error", str(e))
            raise

if __name__ == "__main__":
    args = parse_args()
    main(args)


