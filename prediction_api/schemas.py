from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class TicketRequest(BaseModel):
    subject: str = ""
    description: str = Field(min_length=1)
    customer_gender: str = "Unknown"
    customer_age: int = Field(default=30, ge=1, le=120)
    product_purchased: str = "Unknown"
    purchase_date: date
    ticket_channel: str = "Email"
    satisfaction_rating: float = Field(default=4, ge=1, le=5)


class PredictionResponse(BaseModel):
    category: Optional[str] = None
    category_confidence: float = 0.0
    priority: Optional[str] = None
    resolution_time_hours: Optional[float] = None
    model_version: str = "local"