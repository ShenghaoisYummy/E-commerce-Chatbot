"""
Deployment module for model deployment operations
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

def download_artifacts_from_mlflow(run_id, tracking_uri, tokenizer_artifact_path="tokenizer", adapter_artifact_path="model/peft"):
    """Download tokenizer and adapter artifacts from MLflow"""
    logger.info(f"Setting MLflow tracking URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    
    client = mlflow.tracking.MlflowClient()
    
    # Create temporary directory for downloads
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory: {temp_dir}")
    
    try:
        # Create destination directories
        tokenizer_local_path = os.path.join(temp_dir, "tokenizer_download")
        adapter_local_path = os.path.join(temp_dir, "adapter_download")
        
        # Create the directories
        os.makedirs(tokenizer_local_path, exist_ok=True)
        os.makedirs(adapter_local_path, exist_ok=True)
        
        # Download tokenizer artifacts
        logger.info("Downloading tokenizer artifacts...")
        client.download_artifacts(run_id, tokenizer_artifact_path, tokenizer_local_path)
        
        # Download adapter artifacts  
        logger.info("Downloading adapter artifacts...")
        client.download_artifacts(run_id, adapter_artifact_path, adapter_local_path)
        
        # Find the actual files (MLflow creates subdirectories matching the artifact path)
        tokenizer_files_path = os.path.join(tokenizer_local_path, tokenizer_artifact_path)
        adapter_files_path = os.path.join(adapter_local_path, adapter_artifact_path)
        
        logger.info(f"Looking for tokenizer files at: {tokenizer_files_path}")
        logger.info(f"Looking for adapter files at: {adapter_files_path}")
        
        # Verify files exist and list them
        if os.path.exists(tokenizer_files_path):
            tokenizer_files = os.listdir(tokenizer_files_path)
            logger.info(f"✅ Tokenizer files found: {tokenizer_files}")
        else:
            # Try alternative path structure
            logger.warning(f"❌ Tokenizer path not found at {tokenizer_files_path}")
            logger.info(f"Available directories in {tokenizer_local_path}: {os.listdir(tokenizer_local_path)}")
            # Sometimes the structure might be different, let's explore
            for root, dirs, files in os.walk(tokenizer_local_path):
                if files:
                    logger.info(f"Found files in {root}: {files}")
                    if any(f.endswith('.json') or f.endswith('.model') for f in files):
                        tokenizer_files_path = root
                        logger.info(f"Using tokenizer path: {tokenizer_files_path}")
                        break
            
        if os.path.exists(adapter_files_path):
            adapter_files = os.listdir(adapter_files_path)
            logger.info(f"✅ Adapter files found: {adapter_files}")
        else:
            logger.warning(f"❌ Adapter path not found at {adapter_files_path}")
            logger.info(f"Available directories in {adapter_local_path}: {os.listdir(adapter_local_path)}")
            # Explore the structure
            for root, dirs, files in os.walk(adapter_local_path):
                if files:
                    logger.info(f"Found files in {root}: {files}")
                    if any(f.startswith('adapter_') for f in files):
                        adapter_files_path = root
                        logger.info(f"Using adapter path: {adapter_files_path}")
                        break
            
        return tokenizer_files_path, adapter_files_path, temp_dir
        
    except Exception as e:
        logger.error(f"Error downloading artifacts: {e}")
        shutil.rmtree(temp_dir)
        raise

def load_and_prepare_model(tokenizer_path, adapter_path, base_model_name):
    """Load base model, tokenizer, and apply adapter"""
    logger.info(f"Loading base model: {base_model_name}")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    
    # Load tokenizer from downloaded artifacts
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load adapter
    logger.info(f"Loading LoRA adapter from: {adapter_path}")
    model_with_adapter = PeftModel.from_pretrained(base_model, adapter_path)
    
    return model_with_adapter, tokenizer

def prepare_for_huggingface(model, tokenizer, merge_adapter=True):
    """Prepare model for Hugging Face Hub upload"""
    temp_hf_dir = tempfile.mkdtemp()
    logger.info(f"Preparing model for HF Hub in: {temp_hf_dir}")
    
    if merge_adapter:
        # Merge LoRA adapter with base model for deployment
        logger.info("Merging LoRA adapter with base model...")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(temp_hf_dir)
        tokenizer.save_pretrained(temp_hf_dir)
    else:
        # Save with adapter (recommended for LoRA models)
        model.save_pretrained(temp_hf_dir)
        tokenizer.save_pretrained(temp_hf_dir)
    
    return temp_hf_dir

def create_model_card(model_dir, model_id, base_model_name, performance_metrics=None):
    """Create a model card for Hugging Face Hub"""
    
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

def push_to_huggingface(model_dir, model_id, commit_message="Upload fine-tuned model"):
    """Push model to Hugging Face Hub"""
    logger.info(f"Pushing model to Hugging Face Hub: {model_id}")
    
    try:
        # Create repository (will not fail if it already exists)
        api = HfApi()
        
        logger.info(f"Creating/accessing repository: {model_id}")
        create_repo(model_id, exist_ok=True)
        
        # Upload all files in the model directory
        logger.info("Uploading files...")
        api.upload_folder(
            folder_path=model_dir,
            repo_id=model_id,
            repo_type="model",
            commit_message=commit_message
        )
        
        logger.info(f"✅ Model successfully pushed to: https://huggingface.co/{model_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error pushing to Hugging Face: {e}")
        return False

def deploy_model_to_huggingface(model_info_path, hf_model_id, commit_message="Deploy fine-tuned model", 
                               performance_metrics=None, merge_adapter=True):
    """
    Complete pipeline to deploy model to Hugging Face Hub
    
    Args:
        model_info_path: Path to model info JSON file
        hf_model_id: Hugging Face model ID
        commit_message: Commit message for the upload
        performance_metrics: Optional performance metrics for model card
        merge_adapter: Whether to merge LoRA adapter with base model
        
    Returns:
        Boolean indicating success
    """
    temp_dirs_to_cleanup = []
    
    try:
        # Step 1: Authenticate
        if not authenticate_huggingface():
            logger.error("Failed to authenticate with Hugging Face")
            return False
        
        # Step 2: Load model info
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        
        run_id = model_info.get('mlflow_run_id')
        tracking_uri = model_info.get('tracking_uri')
        base_model_name = model_info.get('model_name')
        
        if not all([run_id, tracking_uri, base_model_name]):
            logger.error("Missing required information in model info file")
            return False
        
        # Step 3: Download artifacts from MLflow
        logger.info("Downloading model artifacts from MLflow...")
        tokenizer_path, adapter_path, download_temp_dir = download_artifacts_from_mlflow(
            run_id, tracking_uri
        )
        temp_dirs_to_cleanup.append(download_temp_dir)
        
        # Step 4: Load and prepare model
        logger.info("Loading and preparing model...")
        model, tokenizer = load_and_prepare_model(tokenizer_path, adapter_path, base_model_name)
        
        # Step 5: Prepare for Hugging Face
        logger.info("Preparing model for Hugging Face Hub...")
        hf_model_dir = prepare_for_huggingface(model, tokenizer, merge_adapter)
        temp_dirs_to_cleanup.append(hf_model_dir)
        
        # Step 6: Create model card
        logger.info("Creating model card...")
        create_model_card(hf_model_dir, hf_model_id, base_model_name, performance_metrics)
        
        # Step 7: Push to Hugging Face
        logger.info("Pushing to Hugging Face Hub...")
        success = push_to_huggingface(hf_model_dir, hf_model_id, commit_message)
        
        if success:
            logger.info(f"✅ Model successfully deployed to: https://huggingface.co/{hf_model_id}")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Error deploying to Hugging Face Hub: {e}")
        return False
        
    finally:
        # Cleanup temporary directories
        for temp_dir in temp_dirs_to_cleanup:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {temp_dir}: {e}")

def test_huggingface_connection():
    """Test connection to Hugging Face Hub"""
    try:
        if authenticate_huggingface():
            api = HfApi()
            user_info = api.whoami()
            logger.info(f"✅ Connected to Hugging Face as: {user_info['name']}")
            return True
        else:
            logger.error("❌ Failed to connect to Hugging Face Hub")
            return False
    except Exception as e:
        logger.error(f"❌ Error testing Hugging Face connection: {e}")
        return False 