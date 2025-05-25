#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import sys
import os
import json
import time
import pandas as pd
import torch
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
                      help='Path to training data reference JSON')
    parser.add_argument('--eval-data', type=str, required=True,
                      help='Path to evaluation data (ChatML JSONL format)')
    parser.add_argument('--n-trials', type=int, default=20,
                      help='Number of trials for optimization')
    parser.add_argument('--n-jobs', type=int, default=1,  # Reduced for stability
                      help='Number of parallel jobs')
    parser.add_argument('--mlflow-uri', type=str, default='https://dagshub.com/ShenghaoisYummy/E-commerce-Chatbot.mlflow',
                      help='MLflow tracking URI')
    return parser.parse_args()

def prepare_eval_data_for_hpo(eval_data_path):
    """
    Convert ChatML JSONL data to format expected by evaluation.
    """
    data = []
    with open(eval_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            text = item['text']
            
            # Extract instruction and response from ChatML format
            parts = text.split('<|user|>')
            if len(parts) < 2:
                continue
                
            user_and_assistant = parts[1]
            user_assistant_parts = user_and_assistant.split('<|assistant|>')
            if len(user_assistant_parts) < 2:
                continue
                
            instruction = user_assistant_parts[0].strip()
            response = user_assistant_parts[1].strip()
            
            data.append({
                'instruction': instruction,
                'response': response,
                'text': text  # Keep original for tokenization
            })
    
    return pd.DataFrame(data)

def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Load model and tokenizer for dataset preparation
    model, tokenizer = load_model_and_tokenizer(
        config['model']['base_model'],
        load_in_8bit=config['model'].get('load_in_8bit', False),
        torch_dtype=getattr(torch, config['model'].get('precision', 'float32')),
        device_map="auto"
    )
    
    # Prepare datasets
    logger.info("Preparing training dataset...")

    try:
        with open(args.train_data, 'r') as f:
            dataset_info = json.load(f)
    except FileNotFoundError:
        logger.error("Training data reference file does not exist. Please run the data preprocessing script first.")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error("Training data reference file content is not valid JSON. Please check or regenerate the file.")
        sys.exit(1)

    # Use dataset path from the reference
    train_dataset_path = dataset_info["dataset_path"]
    
    # Prepare evaluation data
    logger.info("Preparing evaluation dataset...")
    eval_df = prepare_eval_data_for_hpo(args.eval_data)
    
    # Tokenize datasets for training using the same method as fine-tuning
    train_dataset = prepare_dataset(
        train_dataset_path,
        tokenizer,
        max_length=config['training']['max_length'],
        text_column='text'  # Use text column like fine-tuning
    )
    
    # Create a small subset of training data for evaluation during HPO
    eval_tokenized_dataset = prepare_dataset(
        args.eval_data,
        tokenizer,
        max_length=config['training']['max_length'],
        text_column='text'
    )
    
    # Set environment variable for tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Run HPO
    logger.info(f"Starting HPO with {args.n_trials} trials and {args.n_jobs} parallel jobs...")
    results = run_hpo(
        train_dataset=train_dataset,
        eval_tokenized_dataset=eval_tokenized_dataset,
        eval_raw_data=eval_df,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        config=config
    )
    
    if results["best_parameters"] is not None:
        logger.info(f"Best parameters found: {results['best_parameters']}")
        logger.info(f"Best combined score: {results['best_score']}")
    else:
        logger.warning("No successful trials completed. No best parameters found.")

if __name__ == "__main__":
    main() 