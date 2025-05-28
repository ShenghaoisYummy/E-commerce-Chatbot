#!/usr/bin/env python3
# scripts/evaluation_script.py
import sys
import os
import torch
import mlflow
import pandas as pd
import numpy as np
from datetime import datetime
from datasets import Dataset
import json
import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import evaluate_model

# Import modules
from src.fine_tuning import (
    load_model_and_tokenizer,
    generate_response
)
# Import utility modules
from utils.mlflow_utils import (
    mlflow_start_run,
    load_model_from_dagshub
)
from utils.yaml_utils import (
    load_config
)
from utils.system_utils import configure_device_settings

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

def load_chatml_dataset(data_path, eval_size=None):
    """
    Load ChatML dataset from JSONL file and optionally limit size.
    """
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if eval_size and i >= eval_size:
                break
            data.append(json.loads(line.strip()))
    
    df = pd.DataFrame(data)
    return Dataset.from_pandas(df)

def main(args):
    # Load configuration
    config = load_config()
    
    # Generate a run ID and set up directories
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"chatbot_evaluation_{run_timestamp}"
    output_dir = args.output_dir
    
    # Create necessary directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Start MLflow run
    with mlflow_start_run(run_name) as run:
        run_id = run.info.run_id
        print(f"MLflow run ID: {run_id}")
        
        # Configure device settings
        device_config = configure_device_settings(config)
        
        # Check if we should use DagShub for model loading
        use_dagshub = config.get('evaluation', {}).get('use_dagshub', True)
        if args.use_base_model:
            use_dagshub = False
            print("Using base model only (skipping fine-tuned model)")
        else:
            print("Using fine-tuned model")

        model_location_file = args.model_artifact_path
        
        if use_dagshub and os.path.exists(model_location_file):
            try:
                print("Loading model from DagShub/MLflow...")
                # Load adapter from DagShub/MLflow
                model_components = load_model_from_dagshub(model_info_path=model_location_file)
                
                # Get base model name from the model info file
                with open(model_location_file, 'r') as f:
                    model_info = json.load(f)
                    base_model_name = model_info.get('model_name')
                
                if not base_model_name:
                    raise ValueError("Base model name not found in model info file")
                    
                print(f"Loading base model: {base_model_name}")
                
                # The model should already be loaded with the adapter
                if isinstance(model_components["model"], dict):
                    print("Model components keys:", model_components["model"].keys())
                    model = model_components["model"].get("model")
                    tokenizer = model_components["model"].get("tokenizer")
                else:
                    model = model_components["model"]
                    tokenizer = model_components["tokenizer"]
                
                if model is None:
                    raise ValueError("Failed to load model from artifacts")
                    
                # Set model to evaluation mode
                model.eval()
                print("Model loaded successfully!")

            except Exception as e:
                print(f"Error loading model from DagShub/MLflow: {str(e)}")
                print("Falling back to base model...")
                use_dagshub = False
            
        if not use_dagshub:
            # Load base model
            model_name = config.get('model', {}).get('base_model', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
            print(f"Loading base model: {model_name}")
            model, tokenizer = load_model_and_tokenizer(
                model_name,
                load_in_8bit=device_config["use_8bit"],
                torch_dtype=device_config["torch_dtype"],
                device_map=device_config["device_map"]
            )
            mlflow.log_param("model_source", "base_model")
            mlflow.log_param("model_path", model_name)
        
        # Get evaluation dataset
        eval_dataset_path = args.eval_dataset_path
        if not eval_dataset_path:
            print("Evaluation dataset path not specified, exiting.")
            return
            
        print(f"Loading evaluation dataset from {eval_dataset_path}")
        try:
            # Load ChatML dataset
            test_dataset = load_chatml_dataset(eval_dataset_path, args.eval_size)
            print(f"Loaded {len(test_dataset)} samples for evaluation")
        except Exception as e:
            print(f"Error loading evaluation dataset: {e}")
            return
            
        # Evaluate the model
        print("Starting model evaluation...")
        metrics, results_df = evaluate_model(model, tokenizer, test_dataset, config)
        
        # Save metrics to file
        metrics_file = os.path.join(output_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        # Save detailed results
        results_file = os.path.join(output_dir, "evaluation_results.csv")
        results_df.to_csv(results_file, index=False)
        
        # Log metrics to MLflow
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
            
        # Log artifacts
        mlflow.log_artifact(metrics_file)
        mlflow.log_artifact(results_file)
        
        print(f"Evaluation completed successfully.")
        print(f"Metrics saved to {metrics_file}")
        print(f"Detailed results saved to {results_file}")
        
        # Print summary metrics
        print("\nEvaluation Summary:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    args = parse_args()
    main(args)