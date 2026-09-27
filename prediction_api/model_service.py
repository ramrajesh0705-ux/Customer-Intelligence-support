from datetime import datetime
from typing import Any

import joblib
import pandas as pd

from .config import (
    CATEGORY_MODEL_PATH,
    LABEL_ENCODER_PATH,
    PRIORITY_MODEL_PATH,
    PRIORITY_SKOPS_PATH,
    RESOLUTION_MODEL_PATH,
    TOKENIZER_PATH,
)
from .schemas import PredictionResponse, TicketRequest


class ModelService:
    def __init__(self) -> None:
        self.models: dict[str, Any] = {}
        self.errors: dict[str, str] = {}
        self._load_models()

    def _load_models(self) -> None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            if not TOKENIZER_PATH.exists() or not CATEGORY_MODEL_PATH.exists():
                raise FileNotFoundError(f"Category model folder not found: {CATEGORY_MODEL_PATH}")

            self.models["tokenizer"] = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH))
            self.models["category_model"] = AutoModelForSequenceClassification.from_pretrained(
                str(CATEGORY_MODEL_PATH)
            )
            self.models["category_model"].eval()
        except Exception as exc:
            self.errors["category"] = str(exc)

        if LABEL_ENCODER_PATH.exists():
            try:
                self.models["label_encoder"] = joblib.load(str(LABEL_ENCODER_PATH))
            except Exception as exc:
                self.errors["category_label_encoder"] = str(exc)
        elif "category_model" in self.models:
            try:
                self.models["label_encoder"] = self.models["category_model"].config.id2label
            except Exception as exc:
                self.errors["category_label_encoder"] = str(exc)

        try:
            resolution_model = joblib.load(str(RESOLUTION_MODEL_PATH))
            if not hasattr(resolution_model, "predict"):
                raise TypeError("resolution model does not expose predict")
            self.models["resolution_time_predictor"] = resolution_model
        except Exception as exc:
            self.errors["resolution_time"] = str(exc)

        priority_path = PRIORITY_MODEL_PATH if PRIORITY_MODEL_PATH.exists() else PRIORITY_SKOPS_PATH
        if priority_path.exists():
            try:
                if str(priority_path).endswith(".skops"):
                    import skops.io as sio

                    self.models["priority_model"] = sio.load(
                        str(priority_path),
                        trusted=[
                            "sklearn",
                            "numpy",
                            "lightgbm",
                            "collections",
                            "collections.OrderedDict",
                            "lightgbm.basic.Booster",
                            "lightgbm.sklearn.LGBMClassifier",
                            "sklearn.compose._column_transformer._RemainderColsList",
                        ],
                    )
                else:
                    self.models["priority_model"] = joblib.load(str(priority_path))
            except Exception as exc:
                self.errors["priority"] = str(exc)
        else:
            self.errors["priority"] = f"Priority model not found: {priority_path}"

    @property
    def is_ready(self) -> bool:
        return all(
            key in self.models
            for key in (
                "tokenizer",
                "category_model",
                "label_encoder",
                "priority_model",
                "resolution_time_predictor",
            )
        )

    def status(self) -> dict[str, Any]:
        return {
            "ready": self.is_ready,
            "loaded_models": sorted(
                key for key in self.models if key.endswith("model") or key.endswith("predictor")
            ),
            "errors": self.errors,
        }

    @staticmethod
    def _input_frame(ticket: TicketRequest) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Customer Age": ticket.customer_age,
                    "Customer Gender": ticket.customer_gender,
                    "Product Purchased": ticket.product_purchased,
                    "Date of Purchase": ticket.purchase_date,
                    "Ticket Subject": ticket.subject,
                    "Ticket Description": ticket.description,
                    "Ticket Channel": ticket.ticket_channel,
                    "First Response Time": datetime.now().date(),
                    "Customer Satisfaction Rating": ticket.satisfaction_rating,
                }
            ]
        )

    @staticmethod
    def _priority_features(frame: pd.DataFrame, category: str | None) -> pd.DataFrame:
        result = frame.copy()
        result["Ticket Type"] = category or "General Inquiry"
        result["Date of Purchase"] = pd.to_datetime(result["Date of Purchase"], errors="coerce")
        result["First Response Time"] = pd.to_datetime(result["First Response Time"], errors="coerce")
        result["Days Since Purchase"] = (
            result["First Response Time"].max() - result["Date of Purchase"]
        ).dt.days.fillna(0)
        result["Ticket Text"] = result["Ticket Subject"].fillna("") + " " + result["Ticket Description"].fillna("")
        return result[
            [
                "Customer Age",
                "Customer Gender",
                "Product Purchased",
                "Ticket Type",
                "Ticket Channel",
                "Customer Satisfaction Rating",
                "Days Since Purchase",
                "Ticket Text",
            ]
        ]

    @staticmethod
    def _resolution_features(frame: pd.DataFrame, category: str | None, priority: str | None) -> pd.DataFrame:
        result = frame.copy()
        result["Ticket Type"] = category or "General Inquiry"
        result["Ticket Priority"] = priority or "Medium"
        result["Ticket Text"] = result["Ticket Subject"].fillna("") + " " + result["Ticket Description"].fillna("")
        columns = [
            "Customer Gender",
            "Customer Age",
            "Product Purchased",
            "Ticket Type",
            "Ticket Priority",
            "Ticket Channel",
            "Customer Satisfaction Rating",
            "Ticket Text",
        ]
        return result[columns]

    def predict(self, ticket: TicketRequest) -> PredictionResponse:
        frame = self._input_frame(ticket)
        category = None
        confidence = 0.0
        priority = None

        if all(key in self.models for key in ("tokenizer", "category_model")):
            import torch

            text = f"{ticket.subject} {ticket.description}"
            inputs = self.models["tokenizer"](text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                output = self.models["category_model"](**inputs)
            probabilities = torch.softmax(output.logits, dim=1)
            class_id = torch.argmax(probabilities, dim=1).item()

            label_map = self.models.get("label_encoder")
            if isinstance(label_map, dict):
                category = label_map.get(class_id, str(class_id))
            else:
                category = label_map.inverse_transform([class_id])[0]
            confidence = float(probabilities[0][class_id].item())

        if "priority_model" in self.models:
            priority_value = self.models["priority_model"].predict(
                self._priority_features(frame, category)
            )[0]
            priority = str(priority_value)

        resolution_time = None
        if "resolution_time_predictor" in self.models:
            resolution_time = float(
                self.models["resolution_time_predictor"]
                .predict(self._resolution_features(frame, category, priority))[0]
            )

        return PredictionResponse(
            category=str(category) if category is not None else None,
            category_confidence=confidence,
            priority=priority,
            resolution_time_hours=resolution_time,
        )