import os
import torch
import subprocess
from typing import Dict, Any

def get_gpu_memory() -> int:
    """Get available GPU memory in GB"""
    if torch.cuda.is_available():
        try:
            # Test if CUDA is actually functional
            test_tensor = torch.zeros(1, device="cuda")
            del test_tensor  # Clean up
            
            # Now get memory info
        gpu = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(gpu)
        return props.total_memory / 1024**3  # Convert to GB
        except Exception as e:
            print(f"Error getting GPU memory: {e}")
            return 0
    return 0

def check_cuda_libraries():
    """Check if required CUDA libraries are available on the system"""
    required_libs = ["libcudart.so", "libcublas.so", "libcufft.so"]
    missing_libs = []
    
    # Try to locate libraries using ldconfig
    try:
        for lib in required_libs:
            result = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True)
            if lib not in result.stdout:
                missing_libs.append(lib)
    except Exception:
        # If ldconfig fails, try a different approach
        try:
            for lib in required_libs:
                result = subprocess.run(["find", "/usr", "-name", lib + "*"], capture_output=True, text=True)
                if not result.stdout.strip():
                    missing_libs.append(lib)
        except Exception:
            # If both approaches fail, we can't determine library status
            print("Could not check for CUDA libraries, assuming they may be missing")
            return False
    
    if missing_libs:
        print(f"Missing CUDA libraries: {', '.join(missing_libs)}")
        return False
    return True

def configure_device_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Configure device settings based on hardware and user configuration.
    """
    # Get model device settings from config
    device_config = config.get('model', {}).get('device', {})
    use_gpu = device_config.get('use_gpu', True)
    precision = device_config.get('precision', 'float32')
    device_map = device_config.get('device_map', 'auto')
    
    # Check if GPU is reported as available by PyTorch
    gpu_available = torch.cuda.is_available()
    
    # Additional check for CUDA libraries if GPU is reported as available
    cuda_libraries_available = True
    if gpu_available:
        try:
            # Test if CUDA is actually functional by creating a small tensor
            test_tensor = torch.zeros(1, device="cuda")
            del test_tensor  # Clean up
            
            # Check for required CUDA libraries
            cuda_libraries_available = check_cuda_libraries()
            
            # Get GPU information
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = get_gpu_memory()
        print(f"GPU detected: {gpu_name} with {gpu_memory:.2f}GB memory")
            
            if not cuda_libraries_available:
                print("Warning: CUDA libraries may be missing or incompatible")
        except Exception as e:
            print(f"Error initializing CUDA: {e}")
            print("Falling back to CPU despite CUDA being reported as available")
            gpu_available = False
            cuda_libraries_available = False
    
    # Determine device settings
    if use_gpu and gpu_available and cuda_libraries_available:
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