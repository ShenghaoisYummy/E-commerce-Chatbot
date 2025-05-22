import os
import torch
import subprocess
from typing import Dict, Any

def get_gpu_memory() -> int:
    """Get available GPU memory in GB"""
    if torch.cuda.is_available():
        gpu = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(gpu)
        return props.total_memory / 1024**3  # Convert to GB
    return 0

def configure_device_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Configure device settings based on hardware and user configuration.
    """
    # Get model device settings from config
    device_config = config.get('model', {}).get('device', {})
    use_gpu = device_config.get('use_gpu', True)
    precision = device_config.get('precision', 'float32')
    device_map = device_config.get('device_map', 'auto')
    
    # Check GPU availability
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = get_gpu_memory()
        print(f"GPU detected: {gpu_name} with {gpu_memory:.2f}GB memory")
    
    # Determine device settings
    if use_gpu and gpu_available:
        print("Using GPU for training")
        if device_map == "auto":
            device_map = "cuda:0"
        
        # Set dtype based on precision and GPU capabilities
        if precision == "float16" and torch.cuda.is_available():
            torch_dtype = torch.float16
            use_fp16 = True
        elif precision == "bfloat16" and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
            use_fp16 = False
        else:
            torch_dtype = torch.float32
            use_fp16 = False
            
        # Enable 8-bit quantization if specified and enough GPU memory
        use_8bit = config.get('model', {}).get('load_in_8bit', False)
        if use_8bit and gpu_memory < 8:  # Disable 8-bit if less than 8GB VRAM
            print("Warning: Insufficient GPU memory for 8-bit quantization, disabling")
            use_8bit = False
    else:
        print("Using CPU for training")
        device_map = "cpu"
        torch_dtype = torch.float32
        use_fp16 = False
        use_8bit = False
    
    print(f"Using configuration: device={device_map}, dtype={torch_dtype}, 8-bit={use_8bit}, fp16={use_fp16}")
    
    return {
        "device_map": device_map,
        "torch_dtype": torch_dtype,
        "use_8bit": use_8bit,
        "use_fp16": use_fp16
    }