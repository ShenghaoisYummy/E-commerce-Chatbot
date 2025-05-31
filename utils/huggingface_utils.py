"""
Utility functions for Hugging Face Hub operations
"""

import os
import json
import tempfile
import shutil
import logging
from typing import Optional, Dict
import mlflow
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi, create_repo, login
from datetime import datetime

logger = logging.getLogger(__name__)

def authenticate_huggingface():
    """Authenticate with Hugging Face Hub using token from environment"""
    try:
        hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
        if hf_token:
            login(token=hf_token)
            logger.info("✅ Successfully authenticated with Hugging Face Hub")
            return True
        else:
            logger.warning("⚠️ No Hugging Face token found in environment variables")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to authenticate with Hugging Face: {e}")
        return False

def download_model_from_mlflow(model_info_path: str) -> tuple:
    """
    Download model artifacts from MLflow
    
    Args:
        model_info_path: Path to model info JSON file
        
    Returns:
        Tuple of (tokenizer_path, adapter_path, temp_dir)
    """
    try:
        # Load model information
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        
        run_id = model_info.get('mlflow_run_id')
        tracking_uri = model_info.get('tracking_uri')
        
        if not run_id or not tracking_uri:
            raise ValueError("Missing MLflow run ID or tracking URI in model info")
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory: {temp_dir}")
        
        # Download artifacts
        tokenizer_local_path = os.path.join(temp_dir, "tokenizer_download")
        adapter_local_path = os.path.join(temp_dir, "adapter_download")
        
        os.makedirs(tokenizer_local_path, exist_ok=True)
        os.makedirs(adapter_local_path, exist_ok=True)
        
        # Download tokenizer and adapter
        logger.info("Downloading tokenizer artifacts...")
        client.download_artifacts(run_id, "tokenizer", tokenizer_local_path)
        
        logger.info("Downloading adapter artifacts...")
        client.download_artifacts(run_id, "model/peft", adapter_local_path)
        
        # Find actual file paths
        tokenizer_files_path = os.path.join(tokenizer_local_path, "tokenizer")
        adapter_files_path = os.path.join(adapter_local_path, "model", "peft")
        
        # Verify paths exist
        if not os.path.exists(tokenizer_files_path):
            # Try alternative structure
            for root, dirs, files in os.walk(tokenizer_local_path):
                if any(f.endswith('.json') or f.endswith('.model') for f in files):
                    tokenizer_files_path = root
                    break
        
        if not os.path.exists(adapter_files_path):
            # Try alternative structure
            for root, dirs, files in os.walk(adapter_local_path):
                if any(f.startswith('adapter_') for f in files):
                    adapter_files_path = root
                    break
        
        logger.info(f"Tokenizer files at: {tokenizer_files_path}")
        logger.info(f"Adapter files at: {adapter_files_path}")
        
        return tokenizer_files_path, adapter_files_path, temp_dir
        
    except Exception as e:
        logger.error(f"Error downloading model from MLflow: {e}")
        raise

def load_and_merge_model(tokenizer_path: str, adapter_path: str, base_model_name: str):
    """
    Load base model, tokenizer, and apply adapter
    
    Args:
        tokenizer_path: Path to tokenizer files
        adapter_path: Path to adapter files
        base_model_name: Name of the base model
        
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        logger.info(f"Loading base model: {base_model_name}")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load and apply adapter
        logger.info(f"Loading LoRA adapter from: {adapter_path}")
        model_with_adapter = PeftModel.from_pretrained(base_model, adapter_path)
        
        # Merge adapter with base model for deployment
        logger.info("Merging LoRA adapter with base model...")
        merged_model = model_with_adapter.merge_and_unload()
        
        return merged_model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading and merging model: {e}")
        raise

def create_model_card(model_dir: str, model_id: str, base_model_name: str, 
                     performance_metrics: Optional[Dict] = None) -> str:
    """
    Create a model card for the Hugging Face Hub
    
    Args:
        model_dir: Directory where model files are stored
        model_id: Hugging Face model ID
        base_model_name: Name of the base model
        performance_metrics: Optional performance metrics to include
        
    Returns:
        Path to the created model card
    """
    
    # Format performance metrics if provided
    metrics_section = ""
    if performance_metrics:
        metrics_section = f"""
## Performance Metrics

- **Weighted Score**: {performance_metrics.get('current_score', 'N/A'):.4f}
- **BLEU-1**: {performance_metrics.get('bleu1', 'N/A'):.4f}
- **BLEU-2**: {performance_metrics.get('bleu2', 'N/A'):.4f}
- **BLEU-3**: {performance_metrics.get('bleu3', 'N/A'):.4f}
- **BLEU-4**: {performance_metrics.get('bleu4', 'N/A'):.4f}
- **ROUGE-L Precision**: {performance_metrics.get('rougeL_precision', 'N/A'):.4f}
"""
    
    model_card_content = f"""---
license: apache-2.0
base_model: {base_model_name}
tags:
- fine-tuned
- e-commerce
- chatbot
- customer-service
- conversational-ai
language:
- en
pipeline_tag: text-generation
library_name: transformers
---

# {model_id.split('/')[-1]}

This is a fine-tuned version of [{base_model_name}](https://huggingface.co/{base_model_name}) specifically optimized for e-commerce customer service applications.

## Model Details

- **Base Model**: {base_model_name}
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation) with merge
- **Task**: E-commerce customer service conversation
- **Training Data**: ChatML formatted e-commerce conversations
- **Deployment Date**: {datetime.now().strftime('%Y-%m-%d')}

{metrics_section}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("{model_id}")
tokenizer = AutoTokenizer.from_pretrained("{model_id}")

# Generate response
def generate_response(prompt, max_length=150):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs, 
        max_new_tokens=max_length, 
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage for e-commerce
system_prompt = "You are a helpful e-commerce customer service assistant."
user_query = "What's your return policy?"
prompt = f"<|system|>\\n{{system_prompt}}<|im_end|>\\n<|user|>\\n{{user_query}}<|im_end|>\\n<|assistant|>\\n"

response = generate_response(prompt)
print(response)
```

## Training Details

This model was fine-tuned using:
- **LoRA Configuration**: Rank 16, Alpha 32
- **Hyperparameter Optimization**: Optuna-based HPO
- **Experiment Tracking**: MLflow
- **Pipeline Management**: DVC
- **Automated Deployment**: GitHub Actions CI/CD

## Intended Use

This model is specifically designed for:
- E-commerce customer service chatbots
- Product inquiry responses
- Order and shipping assistance
- Return and refund policy explanations
- General shopping assistance

## Limitations

- Optimized for e-commerce scenarios, may not perform well on general conversation
- Responses are based on training data patterns
- May require additional safety filtering for production use
- Performance may vary with out-of-domain queries

## Ethical Considerations

- This model should be used responsibly in customer service applications
- Responses should be monitored for accuracy and appropriateness
- Consider implementing human oversight for complex customer issues

## Citation

```bibtex
@misc{{ecommerce-chatbot-tinyllama,
  title={{Fine-tuned TinyLlama for E-commerce Customer Service}},
  author={{E-commerce Chatbot Team}},
  year={{2024}},
  url={{https://huggingface.co/{model_id}}}
}}
```

## Model Card Contact

For questions about this model, please open an issue in the associated repository.
"""
    
    model_card_path = os.path.join(model_dir, "README.md")
    with open(model_card_path, "w", encoding="utf-8") as f:
        f.write(model_card_content)
    
    logger.info(f"Model card created at: {model_card_path}")
    return model_card_path

def push_to_huggingface_hub(model_info_path: str, 
                           hf_model_id: str,
                           commit_message: str = "Deploy fine-tuned model",
                           performance_metrics: Optional[Dict] = None) -> bool:
    """
    Complete pipeline to push model to Hugging Face Hub
    
    Args:
        model_info_path: Path to model info JSON file
        hf_model_id: Hugging Face model ID
        commit_message: Commit message for the upload
        performance_metrics: Optional performance metrics for model card
        
    Returns:
        Boolean indicating success
    """
    try:
        from src.deployment import deploy_model_to_huggingface
        
        logger.info(f"Starting deployment to Hugging Face Hub: {hf_model_id}")
        
        # Use the deployment module function
        success = deploy_model_to_huggingface(
            model_info_path=model_info_path,
            hf_model_id=hf_model_id,
            commit_message=commit_message,
            performance_metrics=performance_metrics,
            merge_adapter=True  # Merge adapter for deployment
        )
        
        if success:
            logger.info(f"✅ Model successfully deployed to Hugging Face: https://huggingface.co/{hf_model_id}")
        else:
            logger.error("❌ Failed to deploy model to Hugging Face")
        
        return success
        
    except Exception as e:
        logger.error(f"Error deploying to Hugging Face: {e}")
        return False

def test_huggingface_connection() -> bool:
    """Test connection to Hugging Face Hub"""
    try:
        from src.deployment import test_huggingface_connection
        return test_huggingface_connection()
    except Exception as e:
        logger.error(f"❌ Error testing Hugging Face connection: {e}")
        return False

if __name__ == "__main__":
    # Test connection
    test_huggingface_connection() 