import os
import torch
import subprocess

def configure_device_settings(config):
    """
    Configure device settings based on hardware and user configuration.
    """
    # Check if we're in Colab
    is_colab = 'COLAB_GPU' in os.environ
    
    # Choose appropriate dtype based on device and config
    use_8bit = config.get('model', {}).get('load_in_8bit', False)
    use_fp16 = config.get('training', {}).get('fp16', False)
    
    # Force simpler configuration for Colab
    if is_colab:
        print("Google Colab environment detected, using simplified configuration")
        device_map = "auto"
        torch_dtype = torch.float32
        use_8bit = False
        use_fp16 = False
    else:
        # Check if we're on Mac with MPS
        is_mps_available = hasattr(torch, 'mps') and torch.backends.mps.is_available()
        try_cpu = os.environ.get("USE_CPU", "0") == "1"
        
        if try_cpu:
            print("Using CPU as requested by environment variable USE_CPU=1")
            device_map = "cpu"
            torch_dtype = torch.float32
            use_8bit = False
            use_fp16 = False
        elif is_mps_available:
            print("MPS (Apple Silicon) device detected, adapting configuration...")
            # Try to get system info
            try:
                total_ram = subprocess.check_output(['sysctl', '-n', 'hw.memsize']).strip()
                total_ram_gb = int(total_ram) / (1024**3)
                print(f"System has approximately {total_ram_gb:.2f} GB RAM")
                
                # If system has less than 32GB RAM, use CPU
                if total_ram_gb < 32:
                    print("System has less than 32GB RAM, using CPU instead of MPS for stability")
                    device_map = "cpu"
                else:
                    device_map = "mps"
            except:
                # If we can't get system info, default to MPS
                device_map = "mps"
            
            # Use float32 on MPS/CPU as it's more stable
            torch_dtype = torch.float32
            use_8bit = False  # Disable 8-bit on MPS
            use_fp16 = False  # Disable fp16 on MPS
        else:
            device_map = "auto"
            torch_dtype = torch.float16 if use_fp16 else torch.float32
    
    print(f"Using configuration: 8-bit={use_8bit}, fp16={use_fp16}, dtype={torch_dtype}, device_map={device_map}")
    
    return {
        "device_map": device_map,
        "torch_dtype": torch_dtype,
        "use_8bit": use_8bit,
        "use_fp16": use_fp16
    } 