import optuna
import mlflow
import yaml
import os
import concurrent.futures
import logging
from typing import Dict, Any
from pathlib import Path
import torch
import time
from utils.system_utils import configure_device_settings


from utils.mlflow_utils import mlflow_setup_tracking
from src.fine_tuning import (
    load_model_and_tokenizer,
    prepare_model_for_lora,
    prepare_dataset,
    get_training_args,
    get_data_collator,
    get_lora_config,
    update_training_args_from_config
)
from transformers import Trainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config() -> Dict[Any, Any]:
    """Load configuration from params.yaml"""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def calculate_combined_score(metrics: Dict[str, float]) -> float:
    """Calculate combined score from BLEU and ROUGE metrics"""
    bleu_score = metrics.get('bleu4', 0)  # Using BLEU-4 score
    rouge_score = metrics.get('rougeL_fmeasure', 0)  # Using ROUGE-L F-measure
    # Equal weighting between BLEU and ROUGE-L
    return (bleu_score + rouge_score) / 2

def objective(trial: optuna.Trial, base_config: Dict[Any, Any], train_dataset: Any, eval_dataset: Any) -> float:
    """Optuna objective function for hyperparameter optimization"""
    
    # Define hyperparameter search space
    hparams = {
        'lora': {
            'r': trial.suggest_int('lora_r', 8, 32),
            'alpha': trial.suggest_int('lora_alpha', 16, 64),
            'dropout': trial.suggest_float('lora_dropout', 0.05, 0.2)
        },
        'training': {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True),
            'batch_size': trial.suggest_int('batch_size', 1, 4),
            'gradient_accumulation_steps': trial.suggest_int('gradient_accumulation_steps', 1, 4),
            'warmup_steps': trial.suggest_int('warmup_steps', 50, 200)
        }
    }
    
    # Update config with trial parameters
    config = base_config.copy()
    config['lora'].update(hparams['lora'])
    config['training'].update(hparams['training'])
    
    # Configure device settings
    device_settings = configure_device_settings(config)
    
    # Create unique run name and output directory
    run_name = f"hpo_trial_{trial.number}"
    output_dir = f"results/hpo_trial_{trial.number}"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load model and tokenizer with proper device settings
        model, tokenizer = load_model_and_tokenizer(
            config['model']['base_model'],
            load_in_8bit=device_settings['use_8bit'],
            torch_dtype=device_settings['torch_dtype'],
            device_map=device_settings['device_map']
        )
        
        # Prepare model with LoRA
        lora_config = get_lora_config(
            r=config['lora']['r'],
            lora_alpha=config['lora']['alpha'],
            lora_dropout=config['lora']['dropout'],
            target_modules=config['lora']['target_modules']
        )
        model = prepare_model_for_lora(model, lora_config)
        
        # Get training arguments
        training_args = get_training_args(
            output_dir=output_dir,
            num_epochs=config['training']['epochs'],
            batch_size=config['training']['batch_size'],
            gradient_accumulation_steps=config['training']['gradient_accumulation_steps']
        )
        
        # Update training args with device settings
        training_args.fp16 = device_settings['use_fp16']
        update_training_args_from_config(training_args, config)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=get_data_collator(tokenizer)
        )
        
        # Train model
        trainer.train()
        
        # Evaluate model
        metrics = trainer.evaluate()
        
        # Calculate combined score
        combined_score = calculate_combined_score(metrics)
        
        # Log metrics in the parent run
        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.log_params(trial.params)
            mlflow.log_metrics({
                'combined_score': combined_score,
                'bleu4': metrics.get('eval_bleu4', 0),
                'rouge_l_f': metrics.get('eval_rougeL_fmeasure', 0)
            })
        
        return combined_score
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {str(e)}")
        raise optuna.TrialPruned()

def run_hpo(train_dataset: Any, eval_dataset: Any, n_trials: int = 20, n_jobs: int = 4) -> Dict[str, Any]:
    """Run hyperparameter optimization"""
    
    # Load config and setup MLflow
    config = load_config()
    mlflow_setup_tracking(config)
    
    # Create Optuna study
    study_name = f"chatbot_hpo_{int(time.time())}"
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Start parent MLflow run
    with mlflow.start_run(run_name="hpo_experiment") as parent_run:
        try:
            # Run optimization with parallel workers
            study.optimize(
                lambda trial: objective(trial, config, train_dataset, eval_dataset),
                n_trials=n_trials,
                n_jobs=n_jobs,
                timeout=None,
                catch=(Exception,)
            )
            
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if not completed_trials:
                logger.warning("No trials completed successfully.")
                return {
                    "best_parameters": None,
                    "best_score": None
                }
            
            # Get best parameters and score
            best_params = study.best_params
            best_score = study.best_value
            
            # Log best results
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("best_combined_score", best_score)
            
            # Update params.yaml with best parameters
            update_config_with_best_params(best_params)
            
            return {
                "best_parameters": best_params,
                "best_score": best_score
            }
        except Exception as e:
            logger.error(f"HPO failed: {str(e)}")
            raise

def update_config_with_best_params(best_params: Dict[str, Any]) -> None:
    """Update params.yaml with best parameters while preserving structure"""
    # First read the file as text to preserve comments and structure
    with open("params.yaml", "r") as f:
        yaml_content = f.read()
    
    # Load the current config
    config = yaml.safe_load(yaml_content)
    
    # Map best parameters back to config structure
    param_mapping = {
        'lora_r': ('lora', 'r'),
        'lora_alpha': ('lora', 'alpha'),
        'lora_dropout': ('lora', 'dropout'),
        'learning_rate': ('training', 'learning_rate'),
        'weight_decay': ('training', 'weight_decay'),
        'batch_size': ('training', 'batch_size'),
        'gradient_accumulation_steps': ('training', 'gradient_accumulation_steps'),
        'warmup_steps': ('training', 'warmup_steps')
    }
    
    # Update values in the config
    for param_name, value in best_params.items():
        if param_name in param_mapping:
            section, key = param_mapping[param_name]
            config[section][key] = value
    
    # Create a new YAML dumper that preserves the quotes and style
    class PreserveQuotesDumper(yaml.SafeDumper):
        def represent_str(self, data):
            if '\n' in data:
                return self.represent_scalar('tag:yaml.org,2002:str', data, style='|')
            return self.represent_scalar('tag:yaml.org,2002:str', data)
    
    # Update only the values in the original content
    for section in config:
        if isinstance(config[section], dict):
            for key, value in config[section].items():
                # Create the YAML pattern to match
                if isinstance(value, (int, float)):
                    pattern = rf"({key}:\s*)[0-9.-]+(\s*(?:\n|$))"
                    replacement = f"\\g<1>{value}\\2"
                elif isinstance(value, bool):
                    pattern = rf"({key}:\s*)(true|false)(\s*(?:\n|$))"
                    replacement = f"\\g<1>{str(value).lower()}\\3"
                elif isinstance(value, str):
                    pattern = rf'({key}:\s*)["\']?[^"\'\n]*["\']?(\s*(?:\n|$))'
                    replacement = f'\\g<1>{value}\\2'
                elif isinstance(value, list):
                    continue  # Skip lists to preserve their structure
                else:
                    continue
                
                import re
                yaml_content = re.sub(pattern, replacement, yaml_content, flags=re.MULTILINE)
    
    # Write back the modified content
    with open("params.yaml", "w") as f:
        f.write(yaml_content)