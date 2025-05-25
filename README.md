# E-commerce Chatbot

A comprehensive end-to-end machine learning project for building and deploying an intelligent e-commerce chatbot. This project combines modern MLOps practices with a production-ready web application, featuring automated hyperparameter optimization, model fine-tuning, and continuous integration.

## 🚀 Features

- **🤖 Advanced Chatbot**: Fine-tuned TinyLlama model for e-commerce conversations
- **🔄 MLOps Pipeline**: Automated data preprocessing, training, and evaluation with DVC
- **🎯 Hyperparameter Optimization**: Automated HPO using Optuna with MLflow tracking
- **📊 Experiment Tracking**: Comprehensive logging with MLflow and DagHub integration
- **🌐 Web Application**: Modern Next.js chatbot interface with real-time conversations
- **⚡ CI/CD Pipeline**: Automated GitHub Actions workflow for feature development
- **📈 Model Evaluation**: BLEU and ROUGE metrics for response quality assessment

## 🏗️ Architecture

### ML Pipeline

```
Data Preprocessing → Hyperparameter Optimization → Fine-tuning → Evaluation → Deployment
```

### Tech Stack

- **ML Framework**: PyTorch, Transformers, PEFT (LoRA)
- **Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **MLOps**: DVC, MLflow, DagHub
- **HPO**: Optuna
- **Frontend**: Next.js 15, React, TypeScript, Tailwind CSS
- **Database**: PostgreSQL with Drizzle ORM
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
│   ├── evaluation.py             # Model evaluation metrics
│   ├── hpo.py                    # Hyperparameter optimization
│   └── rag_setup.py              # RAG implementation (future)
├── scripts/                       # Executable scripts
│   ├── data_prep_script.py       # Data preprocessing entry point
│   ├── fine_tuning_script.py     # Fine-tuning entry point
│   ├── evaluation_script.py      # Evaluation entry point
│   └── hpo_script.py             # HPO entry point
├── utils/                         # Utility modules
│   ├── mlflow_utils.py           # MLflow integration
│   ├── system_utils.py           # GPU/device utilities
│   └── constants.py              # Project constants
├── data/
│   ├── raw/                      # Original e-commerce datasets
│   ├── processed/                # ChatML formatted JSONL files
│   └── evaluation/               # Test datasets
├── results/                       # Training outputs and evaluations
├── configs/                       # Configuration files
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
export DAGSHUB_USER_TOKEN="your_token"
export MLFLOW_TRACKING_USERNAME="your_username"
export MLFLOW_TRACKING_PASSWORD="your_password"
```

### 3. Run ML Pipeline

#### Option A: Full Automated Pipeline

```bash
# Run complete pipeline with HPO
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
```

### Data Format

The project uses ChatML format for training:

```json
{
  "text": "<|im_start|>system\nYou are a helpful e-commerce assistant.<|im_end|>\n<|im_start|>user\nWhat's the return policy?<|im_end|>\n<|im_start|>assistant\nOur return policy allows returns within 30 days...<|im_end|>"
}
```

## 📊 Experiment Tracking

### MLflow Integration

- **Tracking URI**: https://dagshub.com/ShenghaoisYummy/E-commerce-Chatbot.mlflow
- **Experiments**: Automatic logging of parameters, metrics, and artifacts
- **Model Registry**: Versioned model storage and deployment

### Metrics Tracked

- **Training**: Loss, learning rate, gradient norms
- **Evaluation**: BLEU-4, ROUGE-L, perplexity
- **HPO**: Combined scores, parameter combinations

## 🔄 CI/CD Pipeline

The project includes automated GitHub Actions workflows:

### Feature Development Pipeline

- **Trigger**: Push to feature branches
- **Steps**:
  1. Data preprocessing
  2. Hyperparameter optimization
  3. Model fine-tuning
  4. Evaluation
  5. Results commit and push

### Pipeline Features

- **Automatic parameter updates** from HPO
- **DVC integration** for data versioning
- **MLflow logging** for experiment tracking
- **Artifact management** with cloud storage

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
- **Training parameters**: learning_rate, weight_decay, batch_size
- **Optimization**: Combined BLEU + ROUGE score

## 📈 Model Performance

### Evaluation Metrics

- **BLEU-4**: Measures n-gram overlap with reference responses
- **ROUGE-L**: Evaluates longest common subsequence
- **Combined Score**: Weighted average of BLEU and ROUGE

### Current Results

- Model: TinyLlama-1.1B-Chat-v1.0 with LoRA fine-tuning
- Training: ChatML formatted e-commerce conversations
- Evaluation: Automated metrics with human-readable outputs

## 🌐 Web Application

### Features

- **Real-time chat interface** with streaming responses
- **Conversation history** with persistent storage
- **Modern UI** with dark/light mode support
- **Responsive design** for mobile and desktop
- **Authentication** with NextAuth.js

### Tech Stack

- **Framework**: Next.js 15 with App Router
- **Styling**: Tailwind CSS with Radix UI components
- **Database**: PostgreSQL with Drizzle ORM
- **Deployment**: Vercel with edge functions

## 🚀 Deployment

### Model Deployment

1. **Local**: Direct model loading from checkpoints
2. **Cloud**: MLflow model registry integration
3. **API**: RESTful endpoints for inference

### Web App Deployment

```bash
cd app/ai-chatbot
pnpm build
pnpm start
```

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
- All experiments tracked in MLflow

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TinyLlama** team for the base model
- **Hugging Face** for transformers and datasets
- **MLflow** and **DVC** for MLOps infrastructure
- **Vercel** for deployment platform

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/ShenghaoisYummy/E-commerce-Chatbot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ShenghaoisYummy/E-commerce-Chatbot/discussions)
- **MLflow**: [Experiment Tracking](https://dagshub.com/ShenghaoisYummy/E-commerce-Chatbot.mlflow)

---

**Built with ❤️ for the e-commerce community**
