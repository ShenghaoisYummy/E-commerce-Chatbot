import torch
import pandas as pd
import numpy as np
from datasets import load_dataset
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from rouge_score import rouge_scorer
import nltk

from src.fine_tuning import generate_response

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=False)

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
