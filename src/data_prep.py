import pandas as pd
import numpy as np
import re
from typing import Optional, Dict
import json

class EcommerceDataProcessor:
    """
    Wraps data loading and preprocessing for the Bitext retail e-commerce dataset.
    """

    def __init__(self, file_path: str, sample_size: Optional[int] = None):
        """
        Args:
            file_path: Path to the CSV dataset.
            sample_size: Number of rows to include in the processed dataset (None for all rows).
        """
        self.file_path: str = file_path
        self.sample_size = sample_size
        self.df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """
        Load raw data from CSV into a DataFrame.
        """
        self.df = pd.read_csv(self.file_path)
        
        # Apply sampling if specified
        if self.sample_size is not None and self.sample_size < len(self.df):
            self.df = self.df.sample(n=self.sample_size, random_state=42)
            print(f"Sampled {self.sample_size} rows from dataset")
            
        return self.df

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean a piece of text: lowercase, remove HTML tags, non-alphanumeric chars,
        and collapse whitespace.

        Args:
            text: raw text string
        Returns:
            cleaned text string
        """
        if not isinstance(text, str):
            return ""
        text = text.lower().strip()
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    @staticmethod
    def extract_language_features(tags: str) -> Dict[str, bool]:
        """
        Extract boolean features from the `tags` string.
        """
        features = {
            'is_polite': 'P' in tags,
            'is_colloquial': 'Q' in tags,
            'has_offensive_language': 'W' in tags,
            'has_typos': 'Z' in tags,
            'is_basic_syntax': 'B' in tags,
            'is_question': 'I' in tags,
            'is_complex': 'C' in tags,
            'has_negation': 'N' in tags,
            'has_abbreviations': 'E' in tags,
            'is_keyword_mode': 'K' in tags
        }
        return features

    def preprocess(self) -> pd.DataFrame:
        """
        Preprocess the e-commerce dataset for fine-tuning.
        Keeps only instruction and response, formats as ChatML.

        Returns:
            DataFrame with ChatML formatted conversations
        """
        # Default system prompt for e-commerce chatbot
        DEFAULT_SYSTEM_PROMPT = """You are a helpful e-commerce customer service assistant. Provide accurate, helpful, and friendly responses to customer inquiries about products, orders, shipping, returns, and general shopping assistance."""
        
        def wrap_chatml(row):
            """Convert instruction/response to ChatML format"""
            return (
                "<|system|>\n" +
                DEFAULT_SYSTEM_PROMPT.strip() + "\n" +
                f"<|user|>\n{row['instruction'].strip()}\n" +
                "<|assistant|>\n" +
                f"{row['response'].strip()}\n" +
                "<|end|>"
            )
        
        # Keep only instruction and response columns
        df_processed = self.df[['instruction', 'response']].copy()
        
        # Remove any rows with missing data
        df_processed = df_processed.dropna()
        
        # Create ChatML formatted conversations
        df_processed['chatml'] = df_processed.apply(wrap_chatml, axis=1)
        
        # Keep only the ChatML column for training
        df_final = df_processed[['chatml']].copy()
        df_final.columns = ['text']  # Rename for consistency with training pipeline
        
        return df_final

    def run(self) -> pd.DataFrame:
        """
        Execute the full pipeline: load -> preprocess.

        Returns:
            Final preprocessed DataFrame.
        """
        self.load()
        return self.preprocess()

    def save_as_jsonl(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save DataFrame as JSONL format for training.
        
        Args:
            df: DataFrame to save
            output_path: Path to save JSONL file
        """
        # Convert to JSONL format
        jsonl_path = output_path.replace('.csv', '.jsonl')
        
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                json_line = {"text": row['text']}
                f.write(json.dumps(json_line, ensure_ascii=False) + '\n')
        
        print(f"Saved dataset as JSONL to {jsonl_path}")
        return jsonl_path

    @staticmethod
    def log_dataset_to_mlflow(df, dataset_path, sample_size=None, sample_description=None):
        """
        Log dataset and its metadata to MLflow.
        
        Args:
            df: Pandas DataFrame containing the dataset
            dataset_path: Path where the dataset is saved
            sample_size: Number of rows in sample (if applicable)
            sample_description: Description of the dataset
        
        Returns:
            Dictionary with MLflow run information
        """
        import mlflow
        import os
        import json
        from datetime import datetime
        
        try:
            # Configure MLflow tracking URI from environment
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "https://dagshub.com/ShenghaoisYummy/E-commerce-Chatbot.mlflow")
            mlflow.set_tracking_uri(tracking_uri)
            
            # Set up authentication if credentials are available
            mlflow_username = os.environ.get("MLFLOW_TRACKING_USERNAME")
            mlflow_password = os.environ.get("MLFLOW_TRACKING_PASSWORD")
            dagshub_token = os.environ.get("DAGSHUB_USER_TOKEN")
            
            # Prefer DagHub token if available, otherwise use username/password
            if dagshub_token:
                os.environ["MLFLOW_TRACKING_USERNAME"] = "ShenghaoisYummy"
                os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
            elif mlflow_username and mlflow_password:
                os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
                os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password
            else:
                print("Warning: No MLflow credentials found. Attempting without authentication...")
            
            # Set experiment for datasets specifically
            mlflow.set_experiment("dataset_versions")
            
            # Start MLflow run
            with mlflow.start_run(run_name=f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log dataset parameters
                mlflow.log_param("sample_size", sample_size)
                mlflow.log_param("sample_description", sample_description)
                mlflow.log_param("filename", os.path.basename(dataset_path))
                mlflow.log_param("row_count", len(df))
                mlflow.log_param("format", "jsonl")
                mlflow.log_param("columns", list(df.columns))
                
                # Create dataset reference with metadata
                dataset_ref = {
                    "dataset_path": dataset_path,
                    "dataset_size": len(df),
                    "sample_size": sample_size,
                    "sample_description": sample_description,
                    "format": "ChatML",
                    "columns": list(df.columns),
                    "created_at": datetime.now().isoformat(),
                    "file_size_mb": round(os.path.getsize(dataset_path) / (1024*1024), 2) if os.path.exists(dataset_path) else None
                }
                
                # Log dataset info as JSON artifact (not the dataset file)
                mlflow.log_dict(dataset_ref, "dataset_info.json")
                
                # Log dataset schema separately
                schema_info = {
                    "columns": list(df.columns),
                    "dtypes": {col: str(df[col].dtype) for col in df.columns},
                    "shape": df.shape
                }
                mlflow.log_dict(schema_info, "dataset_schema.json")
                
                # Get run info for reference
                run_id = mlflow.active_run().info.run_id
                
            # Return run information
            return {
                "mlflow_run_id": run_id,
                "tracking_uri": mlflow.get_tracking_uri(),
                "dataset_path": dataset_path
            }
            
        except Exception as e:
            print(f"Error logging to MLflow: {e}")
            print("Continuing without MLflow logging...")
            # Return minimal info for pipeline to continue
            return {
                "mlflow_run_id": None,
                "tracking_uri": tracking_uri,
                "dataset_path": dataset_path,
                "error": str(e)
            }
