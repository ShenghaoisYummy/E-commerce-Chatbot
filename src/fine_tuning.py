# src/fine_tuning.py
import torch
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)

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
        target_modules=target_modules
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
        
        # Load model with optimized settings
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            device_map="cpu",
            use_cache=False if device_map == "cpu" else True,
            low_cpu_mem_usage=True
        )
        model = model.to_empty(device="cuda" if torch.cuda.is_available() else "cpu")

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
    Prepare model for LoRA fine-tuning with GPU optimization.
    """
    try:
        # Handle meta tensors properly
        is_meta = False
        try:
            current_device = next(model.parameters()).device
            # Check if model is using meta tensors
            is_meta = getattr(model, "is_meta", False) or hasattr(next(model.parameters()), "is_meta")
        except Exception as e:
            # If we can't get parameters, it might be a meta tensor model
            if "Cannot access data pointer of Tensor that doesn't have storage" in str(e):
                is_meta = True
                current_device = "meta"
            else:
                print(f"Warning when checking device: {e}")
                current_device = "cpu"
        
        # Prepare model for quantization if needed
        if getattr(model, "is_loaded_in_8bit", False):
            model = prepare_model_for_kbit_training(model)
        
        # Apply LoRA
        try:
            model = get_peft_model(model, lora_config)
        except OSError as e:
            if "libcudart.so" in str(e) and "cannot open shared object file" in str(e):
                print("CUDA library error detected. Moving model to CPU and retrying...")
                # Move model to CPU
                model = model.cpu()
                # Force CPU device map in config
                lora_config.inference_mode = False  # Ensure we're in training mode
                # Retry with CPU
                model = get_peft_model(model, lora_config)
                print("Successfully created PEFT model on CPU")
                return model
            else:
                raise
        
        # Optimize memory usage
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        
        # Handle moving model to proper device
        if torch.cuda.is_available():
            try:
                # Enable gradient checkpointing if available
                if hasattr(model, "gradient_checkpointing_enable"):
                    model.gradient_checkpointing_enable()
                
                # Move model to device, handling meta tensors properly
                if is_meta:
                    try:
                        model = model.to_empty(device="cuda")
                    except AttributeError:
                        # If to_empty not available, try alternative approach
                        print("to_empty not available, using manual device setting")
                        for param in model.parameters():
                            if hasattr(param, "device") and param.device.type == "meta":
                                # Try to initialize meta tensor on device
                                param.data = torch.zeros_like(param, device="cuda")
                elif str(current_device) != "cuda":
                    model = model.to_empty(device="cuda")
            except RuntimeError as e:
                if "CUDA" in str(e) or "cuda" in str(e).lower():
                    print(f"CUDA error when moving model: {e}")
                    print("Falling back to CPU")
                    model = model.cpu()
                else:
                    raise
        
        return model
    except Exception as e:
        print(f"Error preparing model for LoRA: {e}")
        
        # Try to recover by using CPU-only mode
        if "CUDA" in str(e) or "cuda" in str(e).lower() or "libcudart" in str(e):
            print("CUDA error detected, attempting to recover with CPU-only mode...")
            try:
                # Move model to CPU
                model = model.cpu()
                # Apply LoRA
                lora_config.inference_mode = False  # Ensure we're in training mode
                model = get_peft_model(model, lora_config)
                print("Successfully created PEFT model on CPU")
                return model
            except Exception as recovery_error:
                print(f"Recovery attempt failed: {recovery_error}")
        
        raise

def prepare_dataset(data_path, tokenizer, max_length=512, instruction_column="instruction", response_column="response"):
    """
    Load and prepare the dataset for instruction fine-tuning.
    """
    try:
        # Load the preprocessed CSV
        df = pd.read_csv(data_path)
        print(f"Dataset loaded with {len(df)} samples")
        
        # Verify columns exist
        if instruction_column not in df.columns or response_column not in df.columns:
            available_columns = ", ".join(df.columns.tolist())
            raise ValueError(f"Required columns '{instruction_column}' or '{response_column}' not found in dataset. Available columns: {available_columns}")
        
        # Prepare dataset in instruction-following format
        formatted_data = []
        for _, row in df.iterrows():
            # Convert row to dict if it's a Series
            if isinstance(row, pd.Series):
                instruction = row[instruction_column]
                response = row[response_column]
            else:
                instruction = row.get(instruction_column)
                response = row.get(response_column)
                
            formatted_data.append({
                "text": f"### Instruction: {instruction}\n\n### Response: {response}"
            })
        
        dataset = Dataset.from_list(formatted_data)
        
        # Tokenize the dataset with labels for causal language modeling
        def tokenize_function(examples):
            tokenized_inputs = tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors=None  # Don't convert to tensors here
            )
            
            # Create labels - for causal language modeling, labels are the same as input_ids
            tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
            
            return tokenized_inputs
        
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=dataset.column_names
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
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        # GPU optimization settings
        fp16=torch.cuda.is_available(),  # Enable fp16 if GPU available
        bf16=False,  # Disable bfloat16 by default
        gradient_checkpointing=True,  # Enable gradient checkpointing
        optim="adamw_torch",  # Use AdamW optimizer
        learning_rate=2e-4,
        max_grad_norm=1.0,  # Gradient clipping
        # Memory optimization
        dataloader_num_workers=4 if torch.cuda.is_available() else 0,
        dataloader_pin_memory=torch.cuda.is_available(),
    )

def generate_response(instruction, model, tokenizer, max_length=150):
    """
    Generate a response for a given instruction using the fine-tuned model.
    """
    input_text = f"### Instruction: {instruction}\n\n### Response:"
    inputs = tokenizer(input_text, return_tensors="pt").to_empty(device=model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:")[1].strip()

def get_data_collator(tokenizer):
    """
    Create a data collator for language modeling.
    """
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We want causal language modeling, not masked language modeling
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