# Import the processor
import sys
import os
import yaml
from datetime import datetime
import json
import argparse
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_prep import EcommerceDataProcessor

# Load configuration
config_path = "configs/data_prep_config.yaml"
sample_size = None
sample_description = None

def parse_args():
    parser = argparse.ArgumentParser(description="Data preprocessing for ChatML format")
    parser.add_argument("--config", type=str, default=config_path, help="Path to YAML config file")
    parser.add_argument("--input-path", type=str, default="data/raw/ecommerce_chatbot_train.csv", help="Input CSV data path")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--format", type=str, choices=['csv', 'jsonl', 'both'], default='jsonl', help="Output format")
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
            # Get sample size from config (default to None if not specified)
            sample_size = config.get('sampling', {}).get('sample_size', None)
            # Get sample description from config
            sample_description = config.get('sampling', {}).get('sample_description', "")
            # Get input and output paths
            input_path = args.input_path or config.get('data', {}).get('input_path', "data/raw/bitext-retail-ecommerce-llm-chatbot-training-dataset.csv")
            output_dir = args.output_dir or config.get('data', {}).get('output_dir', "data/processed")
        
        print(f"Using configuration from: {config_path}")
        print(f"Sample size: {sample_size}")
        if sample_description:
            print(f"Sample description: {sample_description}")
    except Exception as e:
        print(f"Could not load configuration from {config_path}: {e}")
        print("Using default values")
        input_path = "data/raw/bitext-retail-ecommerce-llm-chatbot-training-dataset.csv"
        output_dir = "data/processed"

    os.makedirs(output_dir, exist_ok=True)

    print("Pre-processing data into ChatML format...")

    # Initialize processor with raw data path
    processor = EcommerceDataProcessor(input_path, sample_size=sample_size)

    # Run the full processing pipeline
    processed_df = processor.run()
    print(f"Processed {len(processed_df)} samples into ChatML format")

    # Generate output filenames
    base_filename = f"chatml_dataset_{sample_description}" if sample_description else "chatml_dataset"
    csv_path = f"{output_dir}/{base_filename}.csv"
    jsonl_path = f"{output_dir}/{base_filename}.jsonl"

    # Save in requested format(s)
    if args.format in ['csv', 'both']:
        processed_df.to_csv(csv_path, index=False)
        print(f"Saved CSV dataset to {csv_path}")
        primary_path = csv_path
    
    if args.format in ['jsonl', 'both']:
        processor.save_as_jsonl(processed_df, csv_path)
        primary_path = jsonl_path

    # Log dataset to MLflow
    try:
        dataset_info = EcommerceDataProcessor.log_dataset_to_mlflow(
            processed_df, 
            primary_path, 
            sample_size=sample_size, 
            sample_description=sample_description
        )
        print("✅ Dataset logged to MLflow successfully")

    except Exception as e:
        print(f"⚠️  MLflow logging failed: {e}")
        print("Continuing with local dataset reference...")
        # Create a local dataset info without MLflow
        dataset_info = {
            "mlflow_run_id": None,
            "tracking_uri": "local",
            "dataset_path": primary_path,
            "sample_size": sample_size,
            "sample_description": sample_description,
            "row_count": len(processed_df),
            "format": "ChatML",
            "error": str(e)
        }

    # Save dataset reference for pipeline
    with open(f"{output_dir}/latest_dataset_ref.json", 'w') as f:
        json.dump(dataset_info, f)

    print(f"Dataset processing completed: {base_filename}")
    print("Sample ChatML format:")
    print(processed_df['text'].iloc[0][:200] + "...")

if __name__ == "__main__":
    main()