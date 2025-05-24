import pandas as pd
import numpy as np
import re
from typing import Optional, Dict

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
        
        # Configure MLflow tracking URI from environment or use default
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "https://dagshub.com/ShenghaoisYummy/E-commerce-Chatbot.mlflow")
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set up authentication if credentials are available
        mlflow_username = os.environ.get("MLFLOW_TRACKING_USERNAME")
        mlflow_password = os.environ.get("MLFLOW_TRACKING_PASSWORD")
        
        if mlflow_username and mlflow_password:
            os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password
        
        # Set experiment for datasets specifically
        mlflow.set_experiment("dataset_versions")
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log dataset parameters
                mlflow.log_param("sample_size", sample_size)
                mlflow.log_param("sample_description", sample_description)
                mlflow.log_param("filename", os.path.basename(dataset_path))
                mlflow.log_param("row_count", len(df))
                mlflow.log_param("column_count", len(df.columns))
                
                # Log dataset statistics
                for column in df.select_dtypes(include=['number']).columns:
                    mlflow.log_metric(f"mean_{column}", df[column].mean())
                    mlflow.log_metric(f"std_{column}", df[column].std())
                
                # Log dataset profile summary
                try:
                    from pandas_profiling import ProfileReport
                    profile = ProfileReport(df, minimal=True, title="Dataset Profile")
                    profile_path = os.path.splitext(dataset_path)[0] + "_profile.html"
                    profile.to_file(profile_path)
                    mlflow.log_artifact(profile_path)
                except ImportError:
                    pass  # Skip if pandas-profiling not available
                
                # Log the dataset file as an artifact
                mlflow.log_artifact(dataset_path)
                
                # Log dataset schema
                schema = {
                    "columns": list(df.columns),
                    "dtypes": {col: str(df[col].dtype) for col in df.columns}
                }
                schema_path = os.path.splitext(dataset_path)[0] + "_schema.json"
                with open(schema_path, 'w') as f:
                    json.dump(schema, f)
                mlflow.log_artifact(schema_path)
                
                # Get run info for reference
                run_id = mlflow.active_run().info.run_id
                
            # Return run information
            return {
                "mlflow_run_id": run_id,
                "tracking_uri": mlflow.get_tracking_uri(),
                "dataset_path": dataset_path
            }

    def preprocess(self) -> pd.DataFrame:
        """
        Preprocess the e-commerce dataset with feature engineering.

        Returns:
            Preprocessed DataFrame with additional features
        """
        df_processed = self.df.copy()

        # Clean instruction and response text
        df_processed['clean_instruction'] = df_processed['instruction'].apply(self.clean_text)
        df_processed['clean_response'] = df_processed['response'].apply(self.clean_text)

        # Extract language features from tags
        language_features = df_processed['tags'].apply(self.extract_language_features)
        df_processed = pd.concat([df_processed, pd.DataFrame(language_features.tolist())], axis=1)

        # Add e-commerce specific features
        df_processed['instruction_length'] = df_processed['clean_instruction'].apply(len)
        df_processed['response_length'] = df_processed['clean_response'].apply(len)
        df_processed['word_count'] = df_processed['clean_instruction'].apply(lambda x: len(x.split()))

        # Create category and intent encodings
        df_processed['category_code'] = pd.Categorical(df_processed['category']).codes
        df_processed['intent_code'] = pd.Categorical(df_processed['intent']).codes

        # Extract question-related features
        df_processed['has_question_mark'] = df_processed['clean_instruction'].apply(lambda x: '?' in x)
        df_processed['starts_with_question_word'] = df_processed['clean_instruction'].apply(
            lambda x: any(x.startswith(w) for w in ['what', 'when', 'where', 'who', 'why', 'how', 'can', 'could', 'would'])
        )

        # Add e-commerce keyword features
        ecommerce_keywords = {
            'price': ['price', 'cost', 'expensive', 'cheap'],
            'shipping': ['shipping', 'delivery', 'ship', 'track'],
            'payment': ['pay', 'payment', 'card', 'refund'],
            'product': ['product', 'item', 'order', 'cart'],
            'return': ['return', 'exchange', 'refund', 'cancel']
        }
        for category, keywords in ecommerce_keywords.items():
            df_processed[f'has_{category}_keywords'] = df_processed['clean_instruction'].apply(
                lambda x, kw=keywords: any(word in x for word in kw)
            )

        # Add urgency indicators
        urgency_words = ['urgent', 'asap', 'emergency', 'immediately', 'quick']
        df_processed['is_urgent'] = df_processed['clean_instruction'].apply(
            lambda x: any(word in x for word in urgency_words)
        )

        # Add sentiment indicators (basic)
        positive_words = ['please', 'thank', 'good', 'great', 'help']
        negative_words = ['bad', 'wrong', 'issue', 'problem', 'error', 'complaint']
        df_processed['has_positive_tone'] = df_processed['clean_instruction'].apply(
            lambda x: any(word in x for word in positive_words)
        )
        df_processed['has_negative_tone'] = df_processed['clean_instruction'].apply(
            lambda x: any(word in x for word in negative_words)
        )

        return df_processed

    def run(self) -> pd.DataFrame:
        """
        Execute the full pipeline: load -> preprocess.

        Returns:
            Final preprocessed DataFrame.
        """
        self.load()
        return self.preprocess()
