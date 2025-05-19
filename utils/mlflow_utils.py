# src/mlflow_utils.py
import mlflow
import os
import json
from datetime import datetime
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec, TensorSpec
import numpy as np
import threading
import time


def mlflow_init(tracking_uri=None, experiment_name=None):
    """
    Set up MLflow tracking.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    if experiment_name:
        mlflow.set_experiment(experiment_name)
        
    return mlflow.get_tracking_uri()

def mlflow_log_model_info(model):
    """
    Log model information to MLflow.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    mlflow.log_param("trainable_params", trainable_params)
    mlflow.log_param("total_params", total_params)
    mlflow.log_param("trainable_percentage", 100 * trainable_params / total_params)

def mlflow_start_run(run_name=None):
    """
    Start a new MLflow run.
    """
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    return mlflow.start_run(run_name=run_name)

def mlflow_setup_tracking(config):
    """
    Set up MLflow tracking and experiment.
    """
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    
    # If not set in environment, use config or default value
    if not mlflow_tracking_uri:
        mlflow_tracking_uri = config.get('mlflow', {}).get('tracking_uri', "file:///./mlruns")
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
        
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", config.get('mlflow', {}).get('experiment_name'))
    
    # Set up MLflow tracking
    tracking_uri = mlflow_init(
        mlflow_tracking_uri,
        experiment_name
    )
    print(f"MLflow tracking URI: {tracking_uri}")
    return tracking_uri

def log_transformers_model(model, tokenizer, task="text-generation", output_dir=None):
    """
    Log PEFT adapter model with detailed progress tracking
    """
    print("Starting adapter logging process...")
    
    try:
        # 1. ensure model is a PEFT model
        if not hasattr(model, 'peft_config'):
            raise ValueError("Model is not a PEFT model. Cannot save adapter weights.")
            
        # 2. save adapter weights
        print("Saving adapter weights...")
        mlflow.transformers.log_model(
            transformers_model={
                "model": model,  
                "tokenizer": tokenizer
            },
            artifact_path="model",
            task=task
        )
        
        # 3. Save tokenizer separately as a dedicated artifact
        print("Saving tokenizer separately...")
        import tempfile, glob
        with tempfile.TemporaryDirectory() as tmp_dir:
            tokenizer.save_pretrained(tmp_dir)
            
            for file_path in glob.glob(os.path.join(tmp_dir, "*")):
                file_name = os.path.basename(file_path)
                mlflow.log_artifact(file_path, "tokenizer")
        
        # 4. record model info
        print("Logging model info...")
        model_info = {
            "base_model": model.config._name_or_path, 
            "fine_tuned": True,
            "task": task,
            "is_adapter": True,
            "adapter_type": "LoRA",
            "tokenizer_saved_separately": True
        }
        mlflow.log_dict(model_info, "model_info.json")

        run_id = mlflow.active_run().info.run_id
        print(f"Adapter and tokenizer logged successfully. Run ID: {run_id}")

        return run_id

    except Exception as e:
        print(f"\nERROR during adapter logging: {str(e)}")
        import traceback
        print(f"Full error traceback:\n{traceback.format_exc()}")
        return None

def load_model_from_dagshub(run_id=None, model_info_path="results/fine_tuned_model_location.json"):
    """
    Load a model from DagShub/MLflow.
    
    Args:
        run_id: Optional explicit run_id to load
        model_info_path: Path to the model info JSON file
        
    Returns:
        model_components: Dictionary containing 'model' and 'tokenizer'
    """
    try:
        # If no run_id provided, try to load from model_info.json
        if not run_id:
            if os.path.exists(model_info_path):
                with open(model_info_path, "r") as f:
                    model_info = json.load(f)
                run_id = model_info.get("mlflow_run_id")
                # If tracking URI is in the file, set it
                if "tracking_uri" in model_info:
                    mlflow.set_tracking_uri(model_info["tracking_uri"])
            else:
                raise ValueError(f"No run_id provided and {model_info_path} not found")
        
        # Construct model URI
        model_uri = f"runs:/{run_id}/transformers-model"
        
        print(f"Loading model from MLflow: {model_uri}")
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        
        # Load the model
        model_components = mlflow.transformers.load_model(model_uri=model_uri)
        
        return model_components
        
    except Exception as e:
        print(f"Error loading model from MLflow: {e}")
        raise