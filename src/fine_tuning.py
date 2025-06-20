# src/fine_tuning.py
import torch
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    BitsAndBytesConfig,
    DataCollatorWithPadding
)
from torch.nn import CrossEntropyLoss

def get_lora_config(r=8, lora_alpha=32, lora_dropout=0.05, target_modules=None):
    """
    Create a LoRA configuration for parameter-efficient fine-tuning.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        inference_mode=False
    )

def load_model_and_tokenizer(model_name, load_in_8bit=False, torch_dtype=torch.float32, device_map="auto"):
    """
    Load pretrained model and tokenizer with optimized GPU settings.
    """
    try:
        # Check if CUDA is actually available despite torch saying it is
        cuda_working = False
        if torch.cuda.is_available():
            try:
                # Test CUDA by creating a small tensor
                test_tensor = torch.zeros(1, device="cuda")
                cuda_working = True
                del test_tensor  # Clean up
            except Exception as e:
                print(f"CUDA reported as available but failed in testing: {e}")
                print("Falling back to CPU")
                device_map = "cpu"
                load_in_8bit = False
                torch_dtype = torch.float32
        
        # Configure quantization if using 8-bit
        if load_in_8bit and cuda_working:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
        else:
            quantization_config = None
        
        # Always load model to meta device first to prevent device copy issues
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            device_map=device_map, 
            use_cache=False if device_map == "cpu" else True,
            low_cpu_mem_usage=True
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
        
    except Exception as e:
        # Handle specific CUDA errors
        error_msg = str(e)
        if "libcudart.so" in error_msg and "cannot open shared object file" in error_msg:
            print("CUDA runtime library error detected. Falling back to CPU.")
            # Retry loading on CPU
            return load_model_and_tokenizer(model_name, load_in_8bit=False, 
                                           torch_dtype=torch.float32, device_map="cpu")
        else:
            # Re-raise other errors
            raise

def prepare_model_for_lora(model, lora_config):
    """
    Prepare model for LoRA fine-tuning with simplified device handling.
    """
    try:
        # Check current device
        try:
            current_device = next(model.parameters()).device
            print(f"Model is currently on device: {current_device}")
        except Exception as e:
            print(f"Could not determine model device: {e}")
            current_device = "cpu"
        
        # Prepare model for quantization if needed
        if getattr(model, "is_loaded_in_8bit", False):
            model = prepare_model_for_kbit_training(model)
        
        # Apply LoRA - this should work regardless of device
        print("Applying LoRA configuration...")
        model = get_peft_model(model, lora_config)
        
        # Optimize memory usage
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        
        # Enable gradient checkpointing if available and on GPU
        if torch.cuda.is_available() and str(current_device) != "cpu":
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
        
        print(f"LoRA model prepared successfully on device: {next(model.parameters()).device}")
        return model
        
    except Exception as e:
        print(f"Error preparing model for LoRA: {e}")
        
        # If there's a meta tensor error, try CPU fallback
        if "meta tensor" in str(e).lower():
            print("Meta tensor error detected, trying CPU fallback...")
            try:
                # Force CPU loading
                model = model.cpu()
                lora_config.inference_mode = False
                model = get_peft_model(model, lora_config)
                print("Successfully created PEFT model on CPU")
                return model
            except Exception as recovery_error:
                print(f"CPU recovery attempt failed: {recovery_error}")
        
        raise

def prepare_dataset(data_path, tokenizer, max_length=512, text_column="text"):
    """
    Load and prepare the ChatML dataset with proper loss masking.
    """
    try:
        # Load the preprocessed dataset (CSV or JSONL)
        if data_path.endswith('.jsonl'):
            import json
            data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            df = pd.DataFrame(data)
        else:
            df = pd.read_csv(data_path)
        
        print(f"Dataset loaded with {len(df)} samples")
        
        # Verify text column exists
        if text_column not in df.columns:
            available_columns = ", ".join(df.columns.tolist())
            raise ValueError(f"Required column '{text_column}' not found in dataset. Available columns: {available_columns}")
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_pandas(df[[text_column]])
        
        def tokenize_with_loss_mask(examples):
            """
            Tokenize with loss masking - only compute loss on assistant responses.
            """
            tokenized_inputs = []
            
            for text in examples[text_column]:
                # Find where assistant response starts
                assistant_idx = text.find("<|assistant|>")
                if assistant_idx == -1:
                    # If no assistant tag found, skip this example
                    print(f"Warning: No <|assistant|> tag found in text: {text[:100]}...")
                    continue
                
                # Add the tag length to get to the actual response start
                assistant_start = assistant_idx + len("<|assistant|>")
                
                # Tokenize the full text with fixed length
                encodings = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",  # Fixed length to avoid tensor issues
                    add_special_tokens=True,
                    return_tensors=None
                )
                
                # Tokenize just the prompt part to determine its length
                prompt_part = text[:assistant_start]
                prompt_tokens = tokenizer(
                    prompt_part, 
                    add_special_tokens=False,  # Include special tokens in prompt
                    return_tensors=None
                )["input_ids"]
                
                # Calculate prompt length (number of tokens to mask)
                prompt_length = len(prompt_tokens)
                
                # Create labels with loss masking
                input_ids = encodings["input_ids"]
                labels = input_ids.copy()
                
                # Mask prompt tokens (set to -100 so they're ignored in loss)
                for i in range(min(prompt_length, len(labels))):
                    labels[i] = -100
                
                # Also mask padding tokens
                for i in range(len(labels)):
                    if input_ids[i] == tokenizer.pad_token_id:
                        labels[i] = -100
                
                tokenized_inputs.append({
                    "input_ids": input_ids,
                    "attention_mask": encodings["attention_mask"],
                    "labels": labels
                })
            
            # Convert list of dicts to dict of lists for HuggingFace datasets
            if not tokenized_inputs:
                return {"input_ids": [], "attention_mask": [], "labels": []}
            
            result = {key: [item[key] for item in tokenized_inputs] for key in tokenized_inputs[0].keys()}
            return result
        
        tokenized_dataset = dataset.map(
            tokenize_with_loss_mask,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing with loss masking"
        )
        
        return tokenized_dataset
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        raise

def get_training_args(output_dir, num_epochs=3, batch_size=8, gradient_accumulation_steps=4):
    """
    Create training arguments with GPU optimization.
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=int(num_epochs),
        per_device_train_batch_size=int(batch_size),
        per_device_eval_batch_size=int(batch_size),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        # GPU optimization settings
        fp16=torch.cuda.is_available(),
        bf16=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        learning_rate=2e-4,
        max_grad_norm=20.0,
        # Memory and data loading optimization
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

def generate_response(instruction, model, tokenizer, max_length=150):
    """
    Generate a response for a given instruction using the ChatML fine-tuned model.
    """
    # Ensure tokenizer has proper pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Format input as ChatML (matching training format)
    DEFAULT_SYSTEM_PROMPT = "You are a helpful e-commerce customer service assistant. Provide accurate, helpful, and friendly responses to customer inquiries about products, orders, shipping, returns, and general shopping assistance."
    
    input_text = (
        "<|system|>\n" +
        DEFAULT_SYSTEM_PROMPT.strip() + "\n" +
        f"<|user|>\n{instruction.strip()}\n" +
        "<|assistant|>\n"
    )
    
    inputs = tokenizer(
        input_text, 
        max_length=512,
        return_tensors="pt", 
        padding="max_length", 
        truncation=True, 
    ).to(model.device)
    
    # Ensure model is in eval mode
    model.eval()
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_length,
                temperature=0.8,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                num_return_sequences=1,
                output_scores=False,
                return_dict_in_generate=False
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant response part
        if "<|assistant|>" in response:
            response_part = response.split("<|assistant|>")[1]
            # Remove <|end|> token if present
            if "<|end|>" in response_part:
                response_part = response_part.split("<|end|>")[0]
            response_part = response_part.strip()
        else:
            response_part = response.strip()
            
        return response_part
        
    except RuntimeError as e:
        if "probability tensor contains" in str(e):
            print(f"Generation failed due to numerical instability: {e}")
            print("Falling back to greedy decoding...")
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=min(max_length, 50),
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        num_return_sequences=1
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if "<|assistant|>" in response:
                    response_part = response.split("<|assistant|>")[1]
                    if "<|end|>" in response_part:
                        response_part = response_part.split("<|end|>")[0]
                    response_part = response_part.strip()
                else:
                    response_part = response.strip()
                    
                return response_part
                
            except Exception as fallback_error:
                print(f"Fallback generation also failed: {fallback_error}")
                return "[Generation failed due to model instability]"
        else:
            raise

def get_data_collator(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token   # 保证有 pad_token

    return DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest",      # 或 "max_length", pad_to_multiple_of=8 等
        return_tensors="pt"
    )

def update_training_args_from_config(training_args, config):
    """
    Update training arguments based on configuration.
    """
    training_config = config.get('training', {})
    if 'learning_rate' in training_config:
        training_args.learning_rate = training_config['learning_rate']
    if 'weight_decay' in training_config:
        training_args.weight_decay = training_config['weight_decay']
    if 'warmup_steps' in training_config:
        training_args.warmup_steps = training_config['warmup_steps']
    if 'fp16' in training_config:
        training_args.fp16 = training_config['fp16']
    if 'evaluation_strategy' in training_config:
        training_args.eval_strategy = training_config['evaluation_strategy']
    if 'save_strategy' in training_config:
        training_args.save_strategy = training_config['save_strategy']
    if 'save_total_limit' in training_config:
        training_args.save_total_limit = training_config['save_total_limit']
    if 'load_best_model_at_end' in training_config:
        training_args.load_best_model_at_end = training_config['load_best_model_at_end']
        
    # Set logging steps from output config if available
    if 'output' in config and 'logging_steps' in config['output']:
        training_args.logging_steps = config['output']['logging_steps']
    
    return training_args

# Only assistant responses contribute to loss
loss = CrossEntropyLoss(ignore_index=-100)
# Tokens with label=-100 are automatically ignored