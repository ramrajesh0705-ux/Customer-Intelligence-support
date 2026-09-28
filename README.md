# Customer Support Intelligence

An AI-powered customer support analytics and prediction system that helps support teams understand ticket trends and estimate how newly submitted tickets should be categorized, prioritized, and resolved. The project combines an interactive Streamlit dashboard with a FastAPI inference service and machine-learning models trained from customer-support ticket data.

## Project Overview & Purpose

Manual ticket triage is slow, inconsistent, and difficult to scale. Customer Support Intelligence turns ticket text and customer/context attributes into actionable predictions that can support routing and service-level decisions.

The system supports three primary prediction tasks:

1. **Ticket category classification** from the subject and description using a Transformer sequence-classification model.
2. **Priority prediction** using a serialized machine-learning model and derived features such as days since purchase and combined ticket text.
3. **Resolution-time estimation** using a serialized regression model and ticket metadata, predicted category, and predicted priority.

It also provides an exploratory analytics dashboard for examining ticket distributions, text patterns, class imbalance, channel performance, resolution time, and customer satisfaction.

## Key Features

- Interactive Streamlit landing page and multi-page dashboard.
- EDA for ticket types, priorities, channels, satisfaction ratings, and resolution times.
- Interactive Plotly charts, Matplotlib/Seaborn visualizations, and ticket-description word clouds.
- Sidebar filters for ticket type, priority, channel, and satisfaction rating.
- Text-length and word-count analysis, including ticket-type comparisons.
- Class-imbalance analysis and ticket-type/priority cross-tab heatmaps.
- FastAPI prediction service with health monitoring at `GET /health`.
- JSON prediction endpoint at `POST /v1/predict`.
- Category confidence scoring for Transformer predictions.
- Graceful partial-model loading: the API records model-loading errors and reports available models through its health response.
- Docker Compose deployment for the frontend and prediction API.
- Training notebooks covering baseline models, classical ML, LSTM, BERT classification, priority, resolution time, customer segmentation, and MLflow viewing.

## Tech Stack

### Runtime and application frameworks

- **Python 3.11**
- **Streamlit** for the interactive dashboard
- **FastAPI** and **Uvicorn** for the prediction API
- **Docker** and **Docker Compose** for containerized execution

### Machine learning and data

- **PyTorch** and **Hugging Face Transformers** for category classification
- **scikit-learn** for preprocessing and classical ML workflows
- **LightGBM-compatible serialized models** for priority prediction where applicable
- **joblib** and **skops** for loading persisted models
- **Pandas** and **NumPy** for data preparation and feature construction

### Visualization and analysis

- **Plotly** for interactive charts
- **Matplotlib** and **Seaborn** for statistical visualizations
- **WordCloud** for ticket-text visualization
- **MLflow** is explored in the training notebooks for experiment tracking/viewing

No external API key is required by the application code currently present. Model files are loaded locally from the configured model paths.

## Architecture / Project Structure

```text
.
├── app/
│   ├── main.py                  # Streamlit home page
│   └── pages/
│       ├── 1_EDA.py             # Interactive exploratory analysis dashboard
│       └── 2_Prediction.py      # Ticket form and API-backed predictions
├── data/
│   └── sample_tickets.csv       # Sample ticket dataset used for analysis/training
├── models/
│   └── category/                # transformer model & encoder
├── resolution_time/model.pkl    # Resolution-time predictor
└── priority/model.skops         # Priority predictor
├── prediction_api/
│   ├── config.py                # Model-root and model-path configuration
│   ├── main.py                  # FastAPI application and HTTP endpoints
│   ├── model_service.py          # Model loading, feature engineering, and inference
│   └── schemas.py               # Pydantic request/response contracts
├── training/
│   ├── baselineModel.ipynb
│   ├── AdvacedClassicalMLModels.ipynb
│   ├── BertClassification.ipynb
│   ├── LSTMClassification.ipynb
│   ├── Predictior.ipynb
│   ├── TicketPriorityBaseLine.ipynb
│   ├── ResolutionTimeandCustomerSegmentaiton .ipynb
│   ├── mlflowuiviewing.ipynb
│   ├── classification_report.txt
│   └── confusion_matrix.png
├── Dockerfile.api               # FastAPI container
├── Dockerfile.frontend          # Streamlit container
├── docker-compose.yml            # API + frontend orchestration
├── requirement.txt               # Legacy/consolidated local requirements
├── requirements-api.txt          # API dependencies
└── requirements-frontend.txt    # Dashboard dependencies
```

### Data flow

1. The EDA page loads `data/sample_tickets.csv` when available, normalizes common column names, derives text and resolution-time fields when needed, and renders filtered analytics.
2. The prediction page collects ticket details and sends them to `PREDICTION_API_URL/v1/predict`.
3. FastAPI validates the payload with `TicketRequest` in `prediction_api/schemas.py`.
4. `ModelService` loads the category Transformer/tokenizer, label encoder, priority model, and resolution-time model from `MODEL_ROOT` or the individual path environment variables.
5. The service builds model-specific Pandas feature frames, runs available models, and returns category, confidence, priority, and estimated resolution time.

## Setup & Installation

### Prerequisites

- Python 3.11 recommended
- Git
- Optional: Docker and Docker Compose
- Trained model artifacts placed in the expected model directories

### 1. Clone the repository

```bash
git clone https://github.com/ramrajesh0705-ux/Customer-Intelligence-support.git
cd Customer-Intelligence-support
```

### 2. Create and activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

For the Streamlit dashboard:

```bash
pip install --upgrade pip
pip install -r requirements-frontend.txt
```

For the prediction API:

```bash
pip install -r requirements-api.txt
pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.14.0+cpu"
```

PyTorch is installed separately from the CPU-only wheel index. This avoids pulling CUDA-enabled wheels and NVIDIA runtime packages on CPU-only machines. For GPU deployments, follow the PyTorch installation instructions for the target CUDA version instead.

For a single local environment, installing both requirement files is the safest option because the dashboard and API are separate processes.

### 4. Add model artifacts

By default, the API searches for the following files and directories:

```text
models/
├── category/                         # Hugging Face/Transformers model directory
├── category/label_encoder.pkl        # Category label encoder
├── resolution_time/model.pkl         # Resolution-time predictor
└── priority/model.skops              # Priority predictor
```

The exact paths can be overridden with environment variables:

```bash
export MODEL_ROOT="$PWD/models"
export CATEGORY_MODEL_PATH="$MODEL_ROOT/category"
export LABEL_ENCODER_PATH="$MODEL_ROOT/category/label_encoder.pkl"
export RESOLUTION_MODEL_PATH="$MODEL_ROOT/resolution_time/model.pkl"
export PRIORITY_MODEL_PATH="$MODEL_ROOT/priority/model.skops"
```

The API can start without every artifact, but `/v1/predict` returns `503` when no prediction model is available, and individual prediction fields remain unavailable when their corresponding model could not be loaded.

### 5. Configure the frontend API URL

For local processes, the prediction page defaults to `http://127.0.0.1:8000`. To use another API location:

```bash
export PREDICTION_API_URL="http://127.0.0.1:8000"
```

There are currently no API keys or third-party credentials required by the application code.

## Usage / Examples

### Run the API locally

From the repository root:

```bash
uvicorn prediction_api.main:app --host 0.0.0.0 --port 8000
```

Check API readiness:

```bash
curl http://127.0.0.1:8000/health
```

Submit a ticket for prediction:

```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Unable to complete payment",
    "description": "My card was charged but the order is still pending.",
    "customer_gender": "Unknown",
    "customer_age": 34,
    "product_purchased": "Premium Plan",
    "purchase_date": "2026-01-15",
    "ticket_channel": "Email",
    "satisfaction_rating": 4
  }'
```

Example response shape:

```json
{
  "category": "Billing inquiry",
  "category_confidence": 0.84,
  "priority": "High",
  "resolution_time_hours": 18.5,
  "model_version": "local"
}
```

Values depend on the model artifacts installed locally.

### Run the Streamlit dashboard

Start the API first, then run:

```bash
streamlit run app/main.py
```

Open the URL shown by Streamlit, normally `http://localhost:8501`. Use the sidebar to open:

- **EDA** for interactive ticket analytics.
- **Prediction** to submit ticket details to the FastAPI service.

### Run both services with Docker Compose

Ensure the model directories are available at `./models`, then run:

```bash
docker compose up --build
```

The services are available at:

- Streamlit dashboard: `http://localhost:8501`
- Prediction API: `http://localhost:8000`
- API health check: `http://localhost:8000/health`

The Compose configuration mounts `./models` read-only into the API container and configures the frontend to call `http://api:8000`.
