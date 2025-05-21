#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import sys
import os
import json
import time
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fine_tuning import prepare_dataset, load_model_and_tokenizer
from src.hpo import run_hpo
from utils.yaml_utils import load_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Run hyperparameter optimization')
    parser.add_argument('--config', type=str, default='params.yaml',
                      help='Path to config file')
    parser.add_argument('--train-data', type=str, required=True,
                      help='Path to training data')
    parser.add_argument('--eval-data', type=str, required=True,
                      help='Path to evaluation data')
    parser.add_argument('--n-trials', type=int, default=20,
                      help='Number of trials for optimization')
    parser.add_argument('--n-jobs', type=int, default=4,
                      help='Number of parallel jobs')
    parser.add_argument('--mlflow-uri', type=str, default='https://dagshub.com/ShenghaoisYummy/E-commerce-Chatbot.mlflow',
                      help='MLflow tracking URI')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Load model and tokenizer for dataset preparation
    model, tokenizer = load_model_and_tokenizer(
        config['model']['base_model'],
        load_in_8bit=config['model'].get('load_in_8bit', True)
    )
    
    # Prepare datasets
    logger.info("Preparing training dataset...")

    try:
        with open(args.train_data, 'r') as f:
            dataset_info = json.load(f)
    except FileNotFoundError:
        print("File does not exist. Please run the data preprocessing script first.")
        # You can return, exit, or handle it in another way
    except json.JSONDecodeError:
        print("File content is not valid JSON. Please check or regenerate the file.")
        # You can return, exit, or handle it in another way

    # Use dataset path from the reference
    train_dataset_path = dataset_info["dataset_path"]

    train_dataset = prepare_dataset(
        train_dataset_path,
        tokenizer,
        max_length=config['sampling']['sample_size'],
        instruction_column=config['data']['instruction_column'],
        response_column=config['data']['response_column']
    )
    
    logger.info("Preparing evaluation dataset...")
    eval_dataset = prepare_dataset(
        args.eval_data,
        tokenizer,
        max_length=config['sampling']['sample_size'],
        instruction_column=config['data']['instruction_column'],
        response_column=config['data']['response_column']
    )
    
    # Run HPO
    logger.info(f"Starting HPO with {args.n_trials} trials and {args.n_jobs} parallel jobs...")
    results = run_hpo(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs
    )
    
    if results["best_parameters"] is not None:
        logger.info(f"Best parameters found: {results['best_parameters']}")
        logger.info(f"Best combined score: {results['best_score']}")
    else:
        logger.warning("No successful trials completed. No best parameters found.")

if __name__ == "__main__":
    main() 