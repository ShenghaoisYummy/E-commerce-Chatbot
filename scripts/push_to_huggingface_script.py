#!/usr/bin/env python3
"""
Script to deploy fine-tuned model to Hugging Face Hub
"""

import os
import sys
import argparse
import logging

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deployment import deploy_model_to_huggingface, test_huggingface_connection
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Deploy model to Hugging Face Hub")
    parser.add_argument(
        "--model-info-path", 
        type=str, 
        default="results/fine_tuned_model_location.json",
        help="Path to model info JSON file"
    )
    parser.add_argument(
        "--hf-model-id", 
        type=str, 
        default="ShenghaoYummy/TinyLlama-ECommerce-Chatbot",
        help="Hugging Face model ID"
    )
    parser.add_argument(
        "--commit-message", 
        type=str, 
        default="Deploy fine-tuned TinyLlama e-commerce chatbot",
        help="Commit message for the upload"
    )
    parser.add_argument(
        "--test-connection", 
        action="store_true",
        help="Test Hugging Face connection and exit"
    )
    parser.add_argument(
        "--no-merge", 
        action="store_true",
        help="Don't merge LoRA adapter with base model"
    )
    
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_args()
    
    logger.info("🚀 Starting Hugging Face deployment process...")
    
    # Test connection if requested
    if args.test_connection:
        logger.info("Testing Hugging Face connection...")
        if test_huggingface_connection():
            logger.info("✅ Connection test successful")
            return 0
        else:
            logger.error("❌ Connection test failed")
            return 1
    
    # Validate inputs
    if not os.path.exists(args.model_info_path):
        logger.error(f"Model info file not found: {args.model_info_path}")
        return 1
    
    if "YourUsername" in args.hf_model_id:
        logger.error("Please update --hf-model-id with your actual Hugging Face username")
        return 1
    
    try:
        # Deploy model
        logger.info(f"Deploying model to: {args.hf_model_id}")
        success = deploy_model_to_huggingface(
            model_info_path=args.model_info_path,
            hf_model_id=args.hf_model_id,
            commit_message=args.commit_message,
            performance_metrics=None,  # Could be loaded from evaluation results if needed
            merge_adapter=not args.no_merge
        )
        
        if success:
            logger.info("✅ Deployment completed successfully!")
            logger.info(f"Your model is now available at: https://huggingface.co/{args.hf_model_id}")
            return 0
        else:
            logger.error("❌ Deployment failed")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Error during deployment: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 