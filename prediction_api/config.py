import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = Path(os.getenv("MODEL_ROOT", PROJECT_ROOT / "models"))
TOKENIZER_PATH = Path(os.getenv("TOKENIZER_PATH", MODEL_ROOT / "category"))
CATEGORY_MODEL_PATH = Path(os.getenv("CATEGORY_MODEL_PATH", MODEL_ROOT / "category" / "model"))
LABEL_ENCODER_PATH = Path(os.getenv("LABEL_ENCODER_PATH", CATEGORY_MODEL_PATH / "label_encoder.pkl"))
RESOLUTION_MODEL_PATH = Path(
    os.getenv("RESOLUTION_MODEL_PATH", MODEL_ROOT / "resolution_time" / "model.pkl")
)
PRIORITY_MODEL_PATH = Path(os.getenv("PRIORITY_MODEL_PATH", MODEL_ROOT / "priority" / "model.pkl"))
API_VERSION = "v1"