#!/usr/bin/env python3
# scripts/evaluation_script.py
import sys
import os
import torch
import mlflow
import pandas as pd
import numpy as np
from datetime import datetime
from datasets import load_dataset
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from rouge_score import rouge_scorer
import nltk
import json
import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from src.fine_tuning import (
    load_model_and_tokenizer,
    generate_response
)
# Import utility modules
from utils.mlflow_utils import (
    mlflow_log_model_info,
    mlflow_start_run,
    mlflow_setup_tracking,
    load_model_from_dagshub
)
from utils.yaml_utils import (
    load_config
)
from utils.constants import RESULTS_DIR
from utils.system_utils import configure_device_settings

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=False)

# Define a simple tokenizer that doesn't rely on punkt_tab
def simple_tokenize(text):
    """
    Simple tokenization function that works without punkt_tab.
    Split by whitespace and punctuation.
    """
    if not isinstance(text, str):
        return []
    # Basic whitespace tokenization
    return text.lower().split()

def calculate_bleu(references, predictions):
    """
    Calculate BLEU score for the given references and predictions.
    
    Args:
        references: List of reference sentences (tokenized)
        predictions: List of model generated sentences (tokenized)
        
    Returns:
        Dict containing BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores
    """
    # For corpus BLEU, we need list of lists (each reference is a list of tokens)
    references_for_corpus = [[ref] for ref in references]
    
    # Calculate BLEU-1 to BLEU-4
    bleu1 = corpus_bleu(
        references_for_corpus, 
        predictions, 
        weights=(1, 0, 0, 0)
    )
    bleu2 = corpus_bleu(
        references_for_corpus, 
        predictions, 
        weights=(0.5, 0.5, 0, 0)
    )
    bleu3 = corpus_bleu(
        references_for_corpus, 
        predictions, 
        weights=(0.33, 0.33, 0.33, 0)
    )
    bleu4 = corpus_bleu(
        references_for_corpus, 
        predictions, 
        weights=(0.25, 0.25, 0.25, 0.25)
    )
    
    return {
        "bleu1": bleu1,
        "bleu2": bleu2,
        "bleu3": bleu3,
        "bleu4": bleu4
    }

def calculate_rouge(references, predictions):
    """
    Calculate ROUGE scores for the given references and predictions.
    
    Args:
        references: List of reference sentences (not tokenized)
        predictions: List of model generated sentences (not tokenized)
        
    Returns:
        Dict containing ROUGE-1, ROUGE-2, and ROUGE-L scores
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    scores = {
        'rouge1_precision': 0.0,
        'rouge1_recall': 0.0,
        'rouge1_fmeasure': 0.0,
        'rouge2_precision': 0.0,
        'rouge2_recall': 0.0,
        'rouge2_fmeasure': 0.0,
        'rougeL_precision': 0.0,
        'rougeL_recall': 0.0,
        'rougeL_fmeasure': 0.0
    }
    
    for ref, hyp in zip(references, predictions):
        rouge_scores = scorer.score(ref, hyp)
        
        # Accumulate scores
        scores['rouge1_precision'] += rouge_scores['rouge1'].precision
        scores['rouge1_recall'] += rouge_scores['rouge1'].recall
        scores['rouge1_fmeasure'] += rouge_scores['rouge1'].fmeasure
        
        scores['rouge2_precision'] += rouge_scores['rouge2'].precision
        scores['rouge2_recall'] += rouge_scores['rouge2'].recall
        scores['rouge2_fmeasure'] += rouge_scores['rouge2'].fmeasure
        
        scores['rougeL_precision'] += rouge_scores['rougeL'].precision
        scores['rougeL_recall'] += rouge_scores['rougeL'].recall
        scores['rougeL_fmeasure'] += rouge_scores['rougeL'].fmeasure
    
    # Calculate average scores
    n = len(references)
    for key in scores:
        scores[key] /= n
    
    return scores

def evaluate_model(model, tokenizer, test_dataset, config):
    """
    Evaluate the model using BLEU and ROUGE metrics.
    
    Args:
        model: The fine-tuned model
        tokenizer: Tokenizer for the model
        test_dataset: Dataset to evaluate on
        config: Configuration dictionary
        
    Returns:
        Dict containing evaluation metrics
    """
    # Get test data
    instructions = test_dataset[config.get('data', {}).get('instruction_column', 'instruction')]
    references = test_dataset[config.get('data', {}).get('response_column', 'response')]
    
    # Generate responses
    predictions = []
    print(f"Generating responses for {len(instructions)} test examples...")
    for i, instruction in enumerate(instructions):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(instructions)}")
        response = generate_response(instruction, model, tokenizer)
        predictions.append(response)
    
    # Save generations for inspection
    results_df = pd.DataFrame({
        'instruction': instructions,
        'reference': references,
        'generated': predictions
    })
    
    # Tokenize for BLEU
    tokenized_references = []
    tokenized_predictions = []
    
    print("Tokenizing text for evaluation...")
    for ref, hyp in zip(references, predictions):
        try:
            # Use simple_tokenize instead of nltk.word_tokenize
            tokenized_ref = simple_tokenize(ref)
        except Exception as e:
            print(f"Error tokenizing reference: {e}")
            tokenized_ref = []
            
        try:
            # Use simple_tokenize instead of nltk.word_tokenize
            tokenized_hyp = simple_tokenize(hyp)
        except Exception as e:
            print(f"Error tokenizing hypothesis: {e}")
            tokenized_hyp = []
            
        tokenized_references.append(tokenized_ref)
        tokenized_predictions.append(tokenized_hyp)
    
    # Calculate metrics
    print("Calculating BLEU scores...")
    bleu_scores = calculate_bleu(tokenized_references, tokenized_predictions)
    
    print("Calculating ROUGE scores...")
    rouge_scores = calculate_rouge(references, predictions)
    
    # Combine metrics
    metrics = {**bleu_scores, **rouge_scores}
    
    return metrics, results_df

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument("--eval-dataset-path", type=str, default="data/evaluation/evaluation_10rows.csv", help="reference location to data path")
    parser.add_argument("--model-artifact-path", type=str, default="results/fine_tuned_model_location.json", help="Path to model artifact file")
    parser.add_argument("--output-dir", type=str, default="results/evaluations", help="Directory to write evaluation results")
    parser.add_argument("--config", type=str, default="params.yaml", help="Path to YAML config file defining evaluation options")
    parser.add_argument("--mlflow-uri", type=str, default="", help="MLflow Tracking Server URI")
    return parser.parse_args()      

def main(args):
    # Load configuration
    config = load_config()
    
    # Set up MLflow tracking
    # tracking_uri = mlflow_setup_tracking(config)
    
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
        model_location_file = args.model_artifact_path
        
        if use_dagshub and os.path.exists(model_location_file):
            try:
                print("Loading model from DagShub/MLflow...")
                # Load adapter from DagShub/MLflow
                model_components = load_model_from_dagshub(model_info_path=model_location_file)
                adapter_path = model_components["model"]
                tokenizer = model_components["tokenizer"]

                # Get base model name from config
                base_model_name = config.get('model', {}).get('base_model')
                if not base_model_name:
                    raise ValueError("Base model name not found in config. Please add 'base_model' under 'model' section.")

                # Load base model and integrate adapter
                try:
                    print(f"Loading base model: {base_model_name}")
                    base_model = AutoModelForCausalLM.from_pretrained(
                        base_model_name,
                        device_map=device_config["device_map"],
                        load_in_8bit=device_config["use_8bit"],
                        torch_dtype=device_config["torch_dtype"]
                    )
                    print("Base model loaded successfully")
                    
                    print(f"Loading adapter from: {adapter_path}")
                    model = PeftModel.from_pretrained(base_model, adapter_path)
                    model.eval()
                    print("Adapter integrated successfully")
                except Exception as e:
                    print(f"Error loading model: {e}")
                    raise

                # Log the model source
                with open(model_location_file, "r") as f:
                    model_info = json.load(f)
                mlflow.log_param("model_source", "dagshub")
                mlflow.log_param("model_run_id", model_info.get("mlflow_run_id", "unknown"))
                
                print("Model loaded successfully from DagShub/MLflow")
            except Exception as e:
                print(f"Error loading model from DagShub/MLflow: {e}")
                print("Falling back to specified model path...")
                use_dagshub = False
        
        if not use_dagshub:
            # Get model path or name from config
            model_name = config.get('evaluation', {}).get('model_path')
            if not model_name:
                print("Error: Model path not specified in config. Add 'model_path' under 'evaluation' section.")
                return
            
            # Load model and tokenizer
            print(f"Loading model from specified path: {model_name}")
            model, tokenizer = load_model_and_tokenizer(
                model_name,
                load_in_8bit=device_config["use_8bit"],
                torch_dtype=device_config["torch_dtype"],
                device_map=device_config["device_map"]
            )
            mlflow.log_param("model_source", "direct_path")
            mlflow.log_param("model_path", model_name)
        
        # Get evaluation dataset
        eval_dataset_path = args.eval_dataset_path
        if not eval_dataset_path:
            print("Evaluation dataset path not specified, exiting.")
            return
            
        print(f"Loading evaluation dataset from {eval_dataset_path}")
        try:
            test_dataset = load_dataset('csv', data_files=eval_dataset_path)['train']
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