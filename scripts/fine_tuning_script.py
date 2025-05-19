#!/usr/bin/env python3
# scripts/fine_tuning_script.py
import sys
import os
import torch
from datetime import datetime
import json
import yaml
import argparse
import mlflow
# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import fine-tuning modules
from src.fine_tuning import (
    get_lora_config, 
    load_model_and_tokenizer, 
    prepare_model_for_lora,
    prepare_dataset, 
    get_training_args,
    get_data_collator,
    update_training_args_from_config,
    generate_response
)

# Import utility modules
from utils.mlflow_utils import (
    mlflow_log_model_info, 
    mlflow_start_run,
    mlflow_setup_tracking,
    log_transformers_model
)
from utils.dvc_utils import setup_environment_data
from utils.yaml_utils import (
    load_config, 
    save_yaml_config, 
    flatten_config
)
from utils.constants import MODELS_DIR
from utils.huggingface_utils import push_to_huggingface_hub
from utils.system_utils import configure_device_settings

from transformers import Trainer
from dotenv import load_dotenv

print("mlflow")
# Load environment variables
load_dotenv()

# Load configuration
config_path = "params.yaml"

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning script")
    parser.add_argument("--config", type=str, default=config_path, help="Path to YAML config file defining cleaning and feature options")
    parser.add_argument("--input-ref", type=str, default="data/processed/latest_dataset_ref.json", help="reference location to data path")
    parser.add_argument("--output-dir", type=str, default="results/", help="Directory to write processed data CSV")
    parser.add_argument("--mlflow-uri", type=str, default="", help="MLflow Tracking Server URI")
    return parser.parse_args()

def main():

    # Load configuration
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    # Set up environment and data
    if not setup_environment_data(config):
        return
    
    # Set up MLflow tracking
    tracking_uri = mlflow_setup_tracking(config)
    
    # Generate a run ID and set up directories
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"chatbot_finetune_{run_timestamp}"
    output_dir = args.output_dir
    
    # Start MLflow run
    with mlflow_start_run() as run:
        run_id = run.info.run_id
        
        # Create necessary directories
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        flattened_params = flatten_config(config)
        mlflow.log_params(flattened_params)

        # Configure device settings
        device_config = configure_device_settings(config)
        
        # Load model and tokenizer
        model_name = config.get('model', {}).get('base_model', "EleutherAI/gpt-neo-125m")
        print(f"Loading base model: {model_name}")
        model, tokenizer = load_model_and_tokenizer(
            model_name,
            load_in_8bit=device_config["use_8bit"],
            torch_dtype=device_config["torch_dtype"],
            device_map=device_config["device_map"]
        )
        
        # Setup LoRA
        lora_config = get_lora_config(
            r=config.get('lora', {}).get('r', 8),
            lora_alpha=config.get('lora', {}).get('alpha', 32),
            lora_dropout=config.get('lora', {}).get('dropout', 0.05),
            target_modules=config.get('lora', {}).get('target_modules')
        )
        model = prepare_model_for_lora(model, lora_config)
        
        # Log model info
        mlflow_log_model_info(model)
        
        # Prepare dataset

        # Load dataset reference
        try:
            with open(args.input_ref, 'r') as f:
                dataset_info = json.load(f)
        except FileNotFoundError:
            print("File does not exist. Please run the data preprocessing script first.")
            # You can return, exit, or handle it in another way
        except json.JSONDecodeError:
            print("File content is not valid JSON. Please check or regenerate the file.")
            # You can return, exit, or handle it in another way

        # Log dataset lineage in the model training run
        mlflow.log_param("dataset_run_id", dataset_info["mlflow_run_id"])
        # Use dataset path from the reference
        dataset_path = dataset_info["dataset_path"]

        if not dataset_path:
            print("Dataset path not specified in config. Exiting.")
            return
            
        tokenized_dataset = prepare_dataset(
            dataset_path, 
            tokenizer, 
            config.get('training', {}).get('max_length', 512),
            config.get('data', {}).get('instruction_column', 'instruction'),
            config.get('data', {}).get('response_column', 'response')
        )
        
        # Split dataset
        split_dataset = tokenized_dataset.train_test_split(
            test_size=config.get('data', {}).get('test_size', 0.1), 
            seed=config.get('data', {}).get('seed', 42)
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        
        # Training arguments
        training_args = get_training_args(
            output_dir=output_dir,
            num_epochs=config.get('training', {}).get('epochs', 3),
            batch_size=config.get('training', {}).get('batch_size', 8),
            gradient_accumulation_steps=config.get('training', {}).get('gradient_accumulation_steps', 4)
        )
        
        # Update training arguments from config
        training_args = update_training_args_from_config(training_args, config)
        
        # Data collator
        data_collator = get_data_collator(tokenizer)
        
        # Define trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )
        
        # Train model
        print("Starting training...")
        train_result = trainer.train()
        
        # Log metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        
        # Evaluate
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        
        # Log all metrics to MLflow
        print("Logging metrics to MLflow...")
        mlflow.log_metrics({f"train_{k}": v for k, v in metrics.items()})
        mlflow.log_metrics({f"eval_{k}": v for k, v in eval_metrics.items()})

        # Log model to MLflow using transformers-specific logging
        print("Logging model to MLflow using transformers-specific logging...")


        # Log the model with transformers-specific logging
        model_run_id = log_transformers_model(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            output_dir=os.path.join(MODELS_DIR, f"transformer_model_{run_timestamp}")
        )
            
            # Save model location information for DVC pipeline
        model_info = {
            "mlflow_run_id": model_run_id or run_id,
            "tracking_uri": mlflow.get_tracking_uri(),
            "dvc_stage": "fine_tuning",  
            "timestamp": run_timestamp,
            "model_name": model_name,
            "fine_tuned": True
        }
            
        with open(os.path.join(output_dir, "fine_tuned_model_location.json"), "w") as f:
            json.dump(model_info, f)
            
        print("Model location information saved to results/fine_tuned_model_location.json")

        # # Push to Hugging Face Hub
        # print("Checking HuggingFace Hub config...")
        # if config.get('hub', {}).get('push_to_hub', False):
        #     print("Attempting to push model to Hugging Face Hub...")
        #     try:
        #         push_to_huggingface_hub(model, tokenizer, config, run_id)
        #     except Exception as e:
        #         print(f"Error pushing to Hugging Face Hub: {e}")
        # else:
        #     print("Skipping Hugging Face Hub push (not enabled in config)")
        
        # print(f"Training completed successfully.")
        # print(f"MLflow run ID: {run_id}")

if __name__ == "__main__":
    args = parse_args()
    main()