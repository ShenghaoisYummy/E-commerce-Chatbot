# Model Deployment Pipeline

This document describes the automated model deployment pipeline that compares model performance and deploys the best performing models to MLflow Model Registry and Hugging Face Hub.

## Overview

The deployment pipeline consists of the following stages:

1. **Evaluation**: Calculate weighted performance scores using BLEU 1-4 and ROUGE-L precision
2. **Comparison**: Compare current model with historical best performance
3. **Decision**: Determine if model should be deployed based on improvement threshold
4. **MLflow Registration**: Register best models in MLflow Model Registry
5. **Hugging Face Deployment**: Push best models to Hugging Face Hub
6. **CI/CD Integration**: Automated deployment through GitHub Actions

## Weighted Scoring System

### Metrics Used

The deployment decision is based on a weighted combination of:

- **BLEU-1** (10%): Basic vocabulary coverage
- **BLEU-2** (15%): Phrase fluency
- **BLEU-3** (25%): Local grammar structure
- **BLEU-4** (40%): Overall quality (most important)
- **ROUGE-L Precision** (10%): Long sequence precision

### Weight Rationale

The weights are designed to prioritize:

1. **Overall Quality**: BLEU-4 gets highest weight (40%) as it's most indicative of complete response quality
2. **Grammar Structure**: BLEU-3 gets second highest (25%) for proper sentence construction
3. **Phrase Fluency**: BLEU-2 (15%) ensures natural word combinations
4. **Precision**: ROUGE-L Precision (10%) prevents information hallucination
5. **Coverage**: BLEU-1 (10%) ensures basic vocabulary relevance

## Deployment Criteria

A model is deployed if:

1. **It's the first model** (no historical data), OR
2. **It achieves the highest weighted score** in history, AND
3. **Improvement exceeds minimum threshold** (default: 0.001)

## Configuration

### DVC Pipeline

The deployment stage is defined in `dvc.yaml`:

```yaml
deployment:
  cmd: >-
    python scripts/deployment_script.py
    --evaluation-results results/evaluations/metrics.json
    --model-artifact-path results/fine_tuned_model_location.json
    --config params.yaml
    --output-dir results/deployment
    --min-improvement-threshold 0.001
    --hf-model-id ShenghaoYummy/TinyLlama-ECommerce-Chatbot
  deps:
    - scripts/deployment_script.py
    - src/model_selection.py
    - results/evaluations/metrics.json
    - results/fine_tuned_model_location.json
    - params.yaml
  params:
    - mlflow.experiment_name
    - mlflow.tracking_uri
  outs:
    - results/deployment
```

### Parameters

Configure deployment in `params.yaml`:

```yaml
deployment:
  min_improvement_threshold: 0.001
  model_registry_name: "ecommerce-chatbot"
  huggingface:
    model_id: "ShenghaoYummy/TinyLlama-ECommerce-Chatbot"
    push_to_hub: true
    create_model_card: true
  weighted_scoring:
    weights:
      bleu1: 0.10
      bleu2: 0.15
      bleu3: 0.25
      bleu4: 0.40
      rougeL_precision: 0.10
```

## Environment Setup

### Required Environment Variables

For GitHub Actions CI/CD:

```bash
# MLflow/DagsHub
DAGSHUB_USER_TOKEN=your_dagshub_token
DAGSHUB_USERNAME=your_username
DAGSHUB_TOKEN=your_dagshub_token
MLFLOW_TRACKING_URI=https://dagshub.com/username/repo.mlflow
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token

# Hugging Face
HF_TOKEN=your_huggingface_token
HUGGINGFACE_HUB_TOKEN=your_huggingface_token

# Optional: AWS for DVC remote storage
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
```

### Local Development

Create a `.env` file:

```bash
HF_TOKEN=your_huggingface_token
DAGSHUB_USER_TOKEN=your_dagshub_token
MLFLOW_TRACKING_URI=https://dagshub.com/username/repo.mlflow
```

## Usage

### Manual Deployment

Run the deployment script manually:

```bash
# Basic deployment
python scripts/deployment_script.py

# With custom parameters
python scripts/deployment_script.py \
  --evaluation-results results/evaluations/metrics.json \
  --model-artifact-path results/fine_tuned_model_location.json \
  --min-improvement-threshold 0.005 \
  --hf-model-id YourUsername/YourModel

# Force deployment (skip performance comparison)
python scripts/deployment_script.py --force-deploy

# Skip Hugging Face deployment
python scripts/deployment_script.py --skip-hf-push
```

### DVC Pipeline

Run the complete pipeline including deployment:

```bash
# Run all stages
dvc repro

# Run only deployment stage
dvc repro deployment
```

### GitHub Actions

The deployment runs automatically in CI/CD:

1. **Automatic**: On push to feature branches
2. **Manual**: Using workflow dispatch with optional force deployment

Trigger manual deployment with force option:

1. Go to Actions tab in GitHub
2. Select "End-to-End ML Pipeline"
3. Click "Run workflow"
4. Check "Force deployment regardless of performance"

## Outputs

### Deployment Results

The deployment stage creates:

```
results/deployment/
├── deployment_info.json      # Deployment decision and status
├── model_comparison.json     # Performance comparison details
└── deployment_logs.txt       # Detailed logs
```

### MLflow Model Registry

Successful deployments create:

- **Registered Model**: Named according to `deployment.model_registry_name`
- **Model Version**: With performance metrics and deployment info
- **Production Stage**: Automatically promoted with archived previous versions

### Hugging Face Hub

Deployed models include:

- **Model Files**: Merged LoRA adapter with base model
- **Tokenizer**: Compatible tokenizer configuration
- **Model Card**: Comprehensive documentation with performance metrics
- **Tags**: Proper categorization for discoverability

## Monitoring

### MLflow Tracking

All deployment runs are tracked in MLflow with:

- **Weighted Score**: Overall performance metric
- **Comparison Results**: Historical performance comparison
- **Deployment Decision**: Whether model was deployed
- **Artifacts**: Comparison results and deployment logs

### GitHub Actions

CI/CD provides:

- **Pipeline Summary**: Complete deployment status
- **Artifacts**: Downloadable deployment results
- **Logs**: Detailed execution logs for debugging

## Troubleshooting

### Common Issues

1. **Authentication Failures**

   ```bash
   # Check tokens
   echo $HF_TOKEN
   echo $DAGSHUB_USER_TOKEN

   # Test connections
   python -c "from utils.huggingface_utils import test_huggingface_connection; test_huggingface_connection()"
   ```

2. **MLflow Connection Issues**

   ```bash
   # Verify MLflow setup
   python -c "import mlflow; print(mlflow.get_tracking_uri())"
   ```

3. **Model Loading Errors**

   ```bash
   # Check model artifacts
   cat results/fine_tuned_model_location.json

   # Verify MLflow run exists
   mlflow runs list --experiment-name ecommerce-chatbot-evaluation
   ```

4. **Deployment Threshold Not Met**
   - Check `deployment_info.json` for comparison details
   - Adjust `min_improvement_threshold` if needed
   - Use `--force-deploy` for testing

### Debug Mode

Enable detailed logging:

```bash
export PYTHONPATH=$PWD
python scripts/deployment_script.py --force-deploy 2>&1 | tee deployment_debug.log
```

## Best Practices

### Performance Thresholds

- **Development**: Use lower thresholds (0.001) for frequent deployments
- **Production**: Use higher thresholds (0.01) for stable deployments
- **A/B Testing**: Deploy multiple models with different thresholds

### Model Versioning

- **Semantic Versioning**: Use meaningful version tags
- **Rollback Strategy**: Keep previous production models accessible
- **Performance Tracking**: Monitor deployed model performance

### Security

- **Token Management**: Use GitHub Secrets for all tokens
- **Access Control**: Limit deployment permissions
- **Audit Trail**: Track all deployments in MLflow

## Integration with Existing Systems

### Web Application

Update your web app to use the latest deployed model:

```python
from huggingface_hub import hf_hub_download
import json

# Get latest model info
model_id = "ShenghaoYummy/TinyLlama-ECommerce-Chatbot"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

### API Endpoints

Create versioned API endpoints:

```python
# app/api/chat/route.ts
const MODEL_ID = process.env.HF_MODEL_ID || "ShenghaoYummy/TinyLlama-ECommerce-Chatbot";
```

### Monitoring

Set up monitoring for deployed models:

- **Performance Metrics**: Track response quality
- **Usage Analytics**: Monitor API usage patterns
- **Error Rates**: Alert on deployment issues

## Future Enhancements

1. **Multi-Model Deployment**: Deploy multiple model variants
2. **Canary Releases**: Gradual rollout of new models
3. **A/B Testing**: Compare model performance in production
4. **Automated Rollback**: Revert to previous version on performance degradation
5. **Custom Metrics**: Add domain-specific evaluation metrics
