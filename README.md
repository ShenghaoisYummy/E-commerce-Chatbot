# E-commerce Chatbot

A comprehensive end-to-end machine learning project for building and deploying an intelligent e-commerce chatbot. This project combines modern MLOps practices with a production-ready web application, featuring automated hyperparameter optimization, model fine-tuning, intelligent model selection, and automated deployment to Hugging Face Hub.

## 🚀 Features

- **🤖 Advanced Chatbot**: Fine-tuned TinyLlama model for e-commerce conversations
- **🔄 MLOps Pipeline**: Automated data preprocessing, training, evaluation, and deployment with DVC
- **🎯 Hyperparameter Optimization**: Automated HPO using Optuna with MLflow tracking
- **📊 Experiment Tracking**: Comprehensive logging with MLflow and DagHub integration
- **🧠 Intelligent Model Selection**: Weighted scoring system using BLEU 1-4 and ROUGE-L metrics
- **🚀 Automated Deployment**: Smart deployment to MLflow Model Registry and Hugging Face Hub
- **🌐 Web Application**: Modern Next.js chatbot interface with real-time conversations
- **⚡ CI/CD Pipeline**: Automated GitHub Actions workflow for feature development
- **📈 Model Evaluation**: Comprehensive BLEU and ROUGE metrics for response quality assessment

## 🏗️ Architecture

### ML Pipeline

```
Data Preprocessing → HPO → Fine-tuning → Evaluation → Model Selection → Deployment
```

### Deployment Pipeline

```
Performance Comparison → Quality Gate → MLflow Registry → Hugging Face Hub
```

### Tech Stack

- **ML Framework**: PyTorch, Transformers, PEFT (LoRA)
- **Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **MLOps**: DVC, MLflow, DagHub
- **HPO**: Optuna
- **Model Selection**: Weighted BLEU/ROUGE scoring
- **Deployment**: MLflow Model Registry, Hugging Face Hub
- **Frontend**: Next.js 15, React, TypeScript, Tailwind CSS
- **Data Storage**: AWS S3
- **Application Database**: PostgreSQL with NEON Serverless Postgres
- **Deployment**: Vercel (Frontend), GitHub Actions (CI/CD)

## 📁 Project Structure

```
E-commerce-Chatbot/
├── app/ai-chatbot/                 # Next.js web application
│   ├── components/                 # React components
│   ├── app/                       # Next.js app router
│   ├── lib/                       # Utilities and database
│   ├── hooks/                     # Custom React hooks
│   └── package.json               # Frontend dependencies
├── src/                           # ML source code
│   ├── data_prep.py              # Data preprocessing with ChatML format
│   ├── fine_tuning.py            # Model fine-tuning with LoRA
│   ├── evaluation.py             # Model evaluation metrics and pipeline
│   ├── model_selection.py        # Weighted scoring and model comparison
│   ├── deployment.py             # Hugging Face deployment functions
│   ├── hpo.py                    # Hyperparameter optimization
│   └── rag_setup.py              # RAG implementation (future)
├── scripts/                       # Executable scripts (argument parsing only)
│   ├── data_prep_script.py       # Data preprocessing entry point
│   ├── fine_tuning_script.py     # Fine-tuning entry point
│   ├── evaluation_script.py      # Evaluation entry point
│   ├── deployment_script.py      # Deployment pipeline entry point
│   ├── push_to_huggingface_script.py # Standalone HF deployment
│   └── hpo_script.py             # HPO entry point
├── utils/                         # Utility modules
│   ├── mlflow_utils.py           # MLflow integration
│   ├── huggingface_utils.py      # Hugging Face utilities
│   ├── system_utils.py           # GPU/device utilities
│   ├── yaml_utils.py             # Configuration utilities
│   └── constants.py              # Project constants
├── data/
│   ├── raw/                      # Original e-commerce datasets
│   ├── processed/                # ChatML formatted JSONL files
│   └── evaluation/               # Test datasets
├── results/                       # Training outputs and evaluations
│   ├── evaluations/              # Model evaluation results
│   └── deployment/               # Deployment decisions and logs
├── configs/                       # Configuration files
├── docs/                         # Documentation
│   └── DEPLOYMENT.md             # Deployment pipeline documentation
├── .github/workflows/             # CI/CD pipeline
├── dvc.yaml                      # DVC pipeline definition
├── params.yaml                   # Model and training parameters
└── requirements.txt              # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+ (for web app)
- CUDA-compatible GPU (recommended)
- Git and DVC

### 1. Clone and Setup

```bash
git clone https://github.com/ShenghaoisYummy/E-commerce-Chatbot.git
cd E-commerce-Chatbot

# Install Python dependencies
pip install -r requirements.txt

# Setup DVC (if using remote storage)
dvc remote add -d storage s3://your-bucket/path
```

### 2. Configure Environment

```bash
# Create .env file with your credentials
# Create .env file with your credentials
export DAGSHUB_USER_TOKEN="your_token"
export MLFLOW_TRACKING_USERNAME="your_username"
export MLFLOW_TRACKING_PASSWORD="your_password"
export HF_TOKEN="your_huggingface_token"  # For deployment

export AWS_ACCESS_KEY_ID="your_aws_access_key"
export AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
export MLFLOW_TRACKING_URI="https://dagshub.com/ShenghaoisYummy/E-commerce-Chatbot.mlflow"

```

### 3. Run ML Pipeline

#### Option A: Full Automated Pipeline with Deployment

```bash
# Run complete pipeline with HPO and smart deployment
dvc repro
```

#### Option B: Step-by-Step Execution

```bash
# 1. Data preprocessing
python scripts/data_prep_script.py

# 2. Hyperparameter optimization (optional)
python scripts/hpo_script.py --n-trials 10

# 3. Model fine-tuning
python scripts/fine_tuning_script.py

# 4. Model evaluation
python scripts/evaluation_script.py

# 5. Smart deployment (only deploys if model improves)
python scripts/deployment_script.py
```

#### Option C: Manual Deployment

```bash
# Deploy to Hugging Face Hub directly
python scripts/push_to_huggingface_script.py --hf-model-id YourUsername/YourModel

# Test Hugging Face connection
python scripts/push_to_huggingface_script.py --test-connection
```

### 4. Launch Web Application

```bash
cd app/ai-chatbot
pnpm install
pnpm dev
```

Visit `http://localhost:3000` to interact with your chatbot!

## 🔧 Configuration

### Model Configuration (`params.yaml`)

```yaml
model:
  base_model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  device:
    use_gpu: true
    precision: "float32"

lora:
  r: 16
  alpha: 32
  dropout: 0.05

training:
  epochs: 2
  batch_size: 4
  learning_rate: 0.0005
  max_length: 512

# New: Deployment configuration
deployment:
  min_improvement_threshold: 0.001
  model_registry_name: "ecommerce-chatbot"
  huggingface:
    model_id: "YourUsername/YourModel"
    push_to_hub: true
  weighted_scoring:
    weights:
      bleu1: 0.10 # Basic vocabulary coverage
      bleu2: 0.15 # Phrase fluency
      bleu3: 0.25 # Local grammar structure
      bleu4: 0.40 # Overall quality (most important)
      rougeL_precision: 0.10 # Content relevance
```

### Data Format

The project uses ChatML format for training:

```json
{
  "text": "<|im_start|>system\nYou are a helpful e-commerce assistant.<|im_end|>\n<|im_start|>user\nWhat's the return policy?<|im_end|>\n<|im_start|>assistant\nOur return policy allows returns within 30 days...<|im_end|>"
}
```

## 🧠 Intelligent Model Selection

### Weighted Scoring System

The deployment pipeline uses a sophisticated weighted scoring system:

- **BLEU-4 (40%)**: Overall response quality and completeness
- **BLEU-3 (25%)**: Local grammar structure and coherence
- **BLEU-2 (15%)**: Phrase fluency and natural combinations
- **BLEU-1 (10%)**: Basic vocabulary coverage
- **ROUGE-L Precision (10%)**: Content relevance and precision

### Deployment Criteria

A model is automatically deployed if:

1. **It's the first model** (no historical data), OR
2. **It achieves the highest weighted score** in history, AND
3. **Improvement exceeds minimum threshold** (configurable, default: 0.001)

### Performance Tracking

```bash
# View model comparison results
cat results/deployment/model_comparison.json

# Check deployment decision
cat results/deployment/deployment_info.json
```

## 📊 Experiment Tracking

### MLflow Integration

- **Tracking URI**: https://dagshub.com/ShenghaoisYummy/E-commerce-Chatbot.mlflow
- **Experiments**: Automatic logging of parameters, metrics, and artifacts
- **Model Registry**: Versioned model storage with automatic promotion to "Production"
- **Deployment Tracking**: Complete audit trail of deployment decisions

### Metrics Tracked

- **Training**: Loss, learning rate, gradient norms
- **Evaluation**: BLEU 1-4, ROUGE-L precision/recall/F1, weighted scores
- **HPO**: Combined scores, parameter combinations
- **Deployment**: Performance comparisons, deployment decisions, model versions

## 🚀 Deployment Pipeline

### Automated Deployment

The deployment pipeline automatically:

1. **Evaluates** the model using comprehensive metrics
2. **Compares** performance with historical best using weighted scoring
3. **Decides** whether to deploy based on improvement thresholds
4. **Registers** best models in MLflow Model Registry
5. **Deploys** to Hugging Face Hub with comprehensive model cards
6. **Tracks** all decisions and performance in MLflow

### Manual Deployment Options

```bash
# Force deployment regardless of performance
python scripts/deployment_script.py --force-deploy

# Skip Hugging Face deployment
python scripts/deployment_script.py --skip-hf-push

# Deploy with custom threshold
python scripts/deployment_script.py --min-improvement-threshold 0.01

# Standalone Hugging Face deployment
python scripts/push_to_huggingface_script.py \
  --hf-model-id YourUsername/YourModel \
  --commit-message "Deploy latest model"
```

## 🔄 CI/CD Pipeline

The project includes automated GitHub Actions workflows:

### Feature Development Pipeline

- **Trigger**: Push to feature branches
- **Steps**:
  1. Data preprocessing
  2. Hyperparameter optimization
  3. Model fine-tuning
  4. Evaluation with weighted scoring
  5. Smart deployment decision
  6. MLflow Model Registry update
  7. Hugging Face Hub deployment (if approved)
  8. Results commit and push

### Pipeline Features

- **Automatic parameter updates** from HPO
- **DVC integration** for data versioning
- **MLflow logging** for experiment tracking
- **Smart deployment** with quality gates
- **Artifact management** with cloud storage
- **Force deployment** option for manual override

## 🎯 Hyperparameter Optimization

### Optuna Integration

```bash
python scripts/hpo_script.py \
  --n-trials 20 \
  --n-jobs 4 \
  --config params.yaml
```

### Search Space

- **LoRA parameters**: r (8-32), alpha (16-64), dropout (0.05-0.2)
- **Training parameters**: learning_rate, weight_decay, batch_size, warmup_steps
- **Optimization**: Weighted BLEU + ROUGE score

### Automatic Parameter Updates

HPO automatically updates `params.yaml` with the best parameters found, which are then used in subsequent training runs.

## 📈 Model Performance

### Evaluation Metrics

- **BLEU 1-4**: Measures n-gram overlap with reference responses
- **ROUGE-L**: Evaluates longest common subsequence precision/recall/F1
- **Weighted Score**: Intelligent combination prioritizing response quality

### Performance Comparison

The system automatically compares:

- Current model vs. historical best
- Improvement percentage and absolute gains
- Contribution breakdown by metric
- Deployment recommendation

### Current Results

- **Model**: TinyLlama-1.1B-Chat-v1.0 with LoRA fine-tuning
- **Training**: ChatML formatted e-commerce conversations
- **Evaluation**: Automated metrics with comprehensive reporting
- **Deployment**: Smart deployment based on performance improvements

## 🌐 Web Application

### Features

- **Real-time chat interface** with streaming responses
- **Conversation history** with persistent storage
- **Modern UI** with dark/light mode support
- **Responsive design** for mobile and desktop
- **Authentication** with NextAuth.js
- **Model integration** with deployed Hugging Face models

### Tech Stack

- **Framework**: Next.js 15 with App Router
- **Styling**: Tailwind CSS with Radix UI components
- **Database**: PostgreSQL with Drizzle ORM
- **Deployment**: Vercel with edge functions

## 🚀 Deployment Options

### Model Deployment

1. **MLflow Model Registry**: Automatic registration and versioning
2. **Hugging Face Hub**: Public model sharing with comprehensive model cards
3. **Local**: Direct model loading from checkpoints
4. **API**: RESTful endpoints for inference

### Web App Deployment

```bash
cd app/ai-chatbot
pnpm build
pnpm start
```

### Environment Variables for Deployment

```bash
# MLflow/DagHub
DAGSHUB_USER_TOKEN=your_dagshub_token
MLFLOW_TRACKING_URI=https://dagshub.com/username/repo.mlflow

# Hugging Face
HF_TOKEN=your_huggingface_token

# Optional: AWS for DVC
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
```

## 📚 Documentation

- **[Deployment Pipeline](docs/DEPLOYMENT.md)**: Comprehensive deployment documentation
- **Model Cards**: Automatically generated on Hugging Face Hub
- **MLflow Experiments**: Detailed experiment tracking and comparison
- **Code Documentation**: Inline documentation and type hints

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Workflow

- Use feature branches for development
- CI/CD pipeline runs automatically on push
- HPO updates parameters automatically
- Smart deployment only deploys improved models
- All experiments tracked in MLflow

### Code Structure

- **Scripts**: Only argument parsing and main execution
- **Modules**: All business logic and complex functions
- **Utils**: Shared utilities and helpers
- **Configs**: Centralized configuration management

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TinyLlama** team for the base model
- **Hugging Face** for transformers, datasets, and Hub
- **MLflow** and **DVC** for MLOps infrastructure
- **Optuna** for hyperparameter optimization
- **Vercel** for deployment platform

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/ShenghaoisYummy/E-commerce-Chatbot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ShenghaoisYummy/E-commerce-Chatbot/discussions)
- **MLflow**: [Experiment Tracking](https://dagshub.com/ShenghaoisYummy/E-commerce-Chatbot.mlflow)
- **Documentation**: [Deployment Guide](docs/DEPLOYMENT.md)

## 🔗 Quick Links

- **🤗 Hugging Face Model**: [TinyLlama-ECommerce-Chatbot](https://huggingface.co/ShenghaoYummy/TinyLlama-ECommerce-Chatbot)
- **📊 MLflow Experiments**: [DagHub MLflow](https://dagshub.com/ShenghaoisYummy/E-commerce-Chatbot.mlflow)
- **🌐 Live Demo**: [Chatbot Web App](https://your-app-url.vercel.app)
- **📖 Deployment Docs**: [DEPLOYMENT.md](docs/DEPLOYMENT.md)

---

**Built with ❤️ for the e-commerce community**
