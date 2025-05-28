#!/usr/bin/env python3
"""
Script to download fine-tuned model artifacts from DagsHub MLflow and push to Hugging Face Hub
"""

import os
import mlflow
import tempfile
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi, create_repo

# ===== CONFIGURATION =====
# Update these values for your specific case
MLFLOW_TRACKING_URI = "https://dagshub.com/ShenghaoisYummy/E-commerce-Chatbot.mlflow"
MLFLOW_RUN_ID = "2b7ff62df540458aa76c68fe1b7798bb"  # Get this from your DagsHub MLflow UI
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
HF_MODEL_ID = "ShenghaoYummy/TinyLlama-ECommerce-Chatbot"  # Your desired HF model name

# Artifact paths from your MLflow (based on the screenshot)
TOKENIZER_ARTIFACT_PATH = "tokenizer"
ADAPTER_ARTIFACT_PATH = "model/peft"

def download_artifacts_from_mlflow(run_id, tracking_uri):
    """Download tokenizer and adapter artifacts from MLflow"""
    print(f"Setting MLflow tracking URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    
    client = mlflow.tracking.MlflowClient()
    
    # Create temporary directory for downloads
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")
    
    try:
        # Create destination directories
        tokenizer_local_path = os.path.join(temp_dir, "tokenizer_download")
        adapter_local_path = os.path.join(temp_dir, "adapter_download")
        
        # Create the directories
        os.makedirs(tokenizer_local_path, exist_ok=True)
        os.makedirs(adapter_local_path, exist_ok=True)
        
        # Download tokenizer artifacts
        print("Downloading tokenizer artifacts...")
        client.download_artifacts(run_id, TOKENIZER_ARTIFACT_PATH, tokenizer_local_path)
        
        # Download adapter artifacts  
        print("Downloading adapter artifacts...")
        client.download_artifacts(run_id, ADAPTER_ARTIFACT_PATH, adapter_local_path)
        
        # Find the actual files (MLflow creates subdirectories matching the artifact path)
        tokenizer_files_path = os.path.join(tokenizer_local_path, TOKENIZER_ARTIFACT_PATH)
        adapter_files_path = os.path.join(adapter_local_path, ADAPTER_ARTIFACT_PATH)
        
        print(f"Looking for tokenizer files at: {tokenizer_files_path}")
        print(f"Looking for adapter files at: {adapter_files_path}")
        
        # Verify files exist and list them
        if os.path.exists(tokenizer_files_path):
            tokenizer_files = os.listdir(tokenizer_files_path)
            print(f"✅ Tokenizer files found: {tokenizer_files}")
        else:
            # Try alternative path structure
            print(f"❌ Tokenizer path not found at {tokenizer_files_path}")
            print(f"Available directories in {tokenizer_local_path}: {os.listdir(tokenizer_local_path)}")
            # Sometimes the structure might be different, let's explore
            for root, dirs, files in os.walk(tokenizer_local_path):
                if files:
                    print(f"Found files in {root}: {files}")
                    if any(f.endswith('.json') or f.endswith('.model') for f in files):
                        tokenizer_files_path = root
                        print(f"Using tokenizer path: {tokenizer_files_path}")
                        break
            
        if os.path.exists(adapter_files_path):
            adapter_files = os.listdir(adapter_files_path)
            print(f"✅ Adapter files found: {adapter_files}")
        else:
            print(f"❌ Adapter path not found at {adapter_files_path}")
            print(f"Available directories in {adapter_local_path}: {os.listdir(adapter_local_path)}")
            # Explore the structure
            for root, dirs, files in os.walk(adapter_local_path):
                if files:
                    print(f"Found files in {root}: {files}")
                    if any(f.startswith('adapter_') for f in files):
                        adapter_files_path = root
                        print(f"Using adapter path: {adapter_files_path}")
                        break
            
        return tokenizer_files_path, adapter_files_path, temp_dir
        
    except Exception as e:
        print(f"Error downloading artifacts: {e}")
        shutil.rmtree(temp_dir)
        raise

def load_and_prepare_model(tokenizer_path, adapter_path):
    """Load base model, tokenizer, and apply adapter"""
    print(f"Loading base model: {BASE_MODEL_NAME}")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
    )
    
    # Load tokenizer from downloaded artifacts
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load adapter
    print(f"Loading LoRA adapter from: {adapter_path}")
    model_with_adapter = PeftModel.from_pretrained(base_model, adapter_path)
    
    return model_with_adapter, tokenizer

def prepare_for_huggingface(model, tokenizer):
    """Prepare model for Hugging Face Hub upload"""
    temp_hf_dir = tempfile.mkdtemp()
    print(f"Preparing model for HF Hub in: {temp_hf_dir}")
    
    # Option 1: Save with adapter (recommended for LoRA models)
    model.save_pretrained(temp_hf_dir)
    tokenizer.save_pretrained(temp_hf_dir)
    
    # Option 2: If you want to merge and save as a single model (uncomment below)
    # print("Merging LoRA adapter with base model...")
    # merged_model = model.merge_and_unload()
    # merged_model.save_pretrained(temp_hf_dir)
    # tokenizer.save_pretrained(temp_hf_dir)
    
    return temp_hf_dir

def push_to_huggingface(model_dir, model_id):
    """Push model to Hugging Face Hub"""
    print(f"Pushing model to Hugging Face Hub: {model_id}")
    
    try:
        # Create repository (will not fail if it already exists)
        api = HfApi()
        
        print(f"Creating/accessing repository: {model_id}")
        create_repo(model_id, exist_ok=True)
        
        # Upload all files in the model directory
        print("Uploading files...")
        api.upload_folder(
            folder_path=model_dir,
            repo_id=model_id,
            repo_type="model",
            commit_message="Upload fine-tuned TinyLlama e-commerce chatbot with LoRA adapter"
        )
        
        print(f"✅ Model successfully pushed to: https://huggingface.co/{model_id}")
        
    except Exception as e:
        print(f"Error pushing to Hugging Face: {e}")
        raise

def create_model_card(model_dir, model_id):
    """Create a basic model card"""
    model_card_content = f"""---
license: apache-2.0
base_model: {BASE_MODEL_NAME}
tags:
- fine-tuned
- e-commerce
- chatbot
- lora
- peft
language:
- en
pipeline_tag: text-generation
---

# {model_id.split('/')[-1]}

This is a fine-tuned version of [{BASE_MODEL_NAME}](https://huggingface.co/{BASE_MODEL_NAME}) for e-commerce customer service chatbot applications.

## Model Details

- **Base Model**: {BASE_MODEL_NAME}
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Task**: E-commerce customer service conversation
- **Training Data**: ChatML formatted e-commerce conversations

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained("{BASE_MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained("{model_id}")

# Load the fine-tuned model
model = PeftModel.from_pretrained(base_model, "{model_id}")

# Generate response
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.8)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
prompt = "<|system|>\\nYou are a helpful e-commerce assistant.<|im_end|>\\n<|user|>\\nWhat's your return policy?<|im_end|>\\n<|assistant|>\\n"
response = generate_response(prompt)
print(response)
```

## Training Details

This model was fine-tuned using:
- LoRA with rank 16
- Learning rate optimization via Optuna
- MLflow experiment tracking
- DVC pipeline management

## Limitations

- Optimized for e-commerce customer service scenarios
- May not perform well on general conversation topics
- Responses are based on training data patterns

## Citation

If you use this model, please cite: @misc{{ecommerce-chatbot-tinyllama,
title={{Fine-tuned TinyLlama for E-commerce Customer Service}},
author={{Your Name}},
year={{2024}},
url={{https://huggingface.co/{model_id}}}
}}"""
    
    model_card_path = os.path.join(model_dir, "README.md")
    with open(model_card_path, "w", encoding="utf-8") as f:
        f.write(model_card_content)
    
    print(f"Model card created at: {model_card_path}")

def main():
    """Main execution function"""
    print("🚀 Starting MLflow to Hugging Face transfer process...")
    
    # Validate configuration
    if MLFLOW_RUN_ID == "YOUR_RUN_ID_HERE":
        print("❌ Please update MLFLOW_RUN_ID in the configuration section")
        return
    
    if "YourUsername" in HF_MODEL_ID:
        print("❌ Please update HF_MODEL_ID with your actual Hugging Face username")
        return
    
    temp_dirs_to_cleanup = []
    
    try:
        # Step 1: Download artifacts from MLflow
        print("\n📥 Step 1: Downloading artifacts from DagsHub MLflow...")
        tokenizer_path, adapter_path, download_temp_dir = download_artifacts_from_mlflow(
            MLFLOW_RUN_ID, MLFLOW_TRACKING_URI
        )
        temp_dirs_to_cleanup.append(download_temp_dir)
        
        # Step 2: Load model and adapter
        print("\n🔧 Step 2: Loading model and applying adapter...")
        model, tokenizer = load_and_prepare_model(tokenizer_path, adapter_path)
        
        # Step 3: Prepare for Hugging Face
        print("\n📦 Step 3: Preparing model for Hugging Face Hub...")
        hf_model_dir = prepare_for_huggingface(model, tokenizer)
        temp_dirs_to_cleanup.append(hf_model_dir)
        
        # Step 4: Create model card
        print("\n📝 Step 4: Creating model card...")
        create_model_card(hf_model_dir, HF_MODEL_ID)
        
        # Step 5: Push to Hugging Face
        print("\n🚀 Step 5: Pushing to Hugging Face Hub...")
        push_to_huggingface(hf_model_dir, HF_MODEL_ID)
        
        print("\n✅ Process completed successfully!")
        print(f"Your model is now available at: https://huggingface.co/{HF_MODEL_ID}")
        
    except Exception as e:
        print(f"\n❌ Error during process: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup temporary directories
        print("\n🧹 Cleaning up temporary files...")
        for temp_dir in temp_dirs_to_cleanup:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"Cleaned up: {temp_dir}")

if __name__ == "__main__":
    main()