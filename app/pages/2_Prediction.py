import json
import os
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st


API_URL = os.getenv("PREDICTION_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="Ticket Intelligence Predictor",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; font-weight: 500; color: #2c3e50; margin-bottom: 0.5rem; }
    .prediction-card { background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #1f77b4; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    </style>
    """,
    unsafe_allow_html=True,
)


def call_api(path: str, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if body else {}
    request = Request(f"{API_URL}{path}", data=body, headers=headers, method="POST" if body else "GET")
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def display_prediction(prediction: dict) -> None:
    st.markdown("#### 🏷️ Category")
    if prediction.get("category"):
        st.markdown(
            f'<div class="prediction-card"><strong>{prediction["category"]}</strong><br>'
            f'<small>Confidence: {prediction.get("category_confidence", 0):.1%}</small></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Category prediction is not available.")

    st.markdown("#### ⚡ Priority")
    if prediction.get("priority"):
        st.markdown(
            f'<div class="prediction-card"><strong>{prediction["priority"]}</strong></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Priority prediction is not available.")

    st.markdown("#### ⏱️ Resolution Time")
    hours = prediction.get("resolution_time_hours")
    if isinstance(hours, (int, float)):
        display_time = f"{hours:.1f} hours" if hours < 24 else f"{hours / 24:.1f} days ({hours:.1f} hours)"
        st.markdown(
            f'<div class="prediction-card"><strong>{display_time}</strong></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Resolution-time prediction is not available.")


st.markdown('<p class="main-header">🎫 Ticket Intelligence Predictor</p>', unsafe_allow_html=True)
st.markdown("Predict ticket category, priority, and estimated resolution time using the prediction API.")
st.divider()

with st.sidebar:
    st.markdown("### 📊 About")
    st.info(
        "This page sends ticket details to the local FastAPI prediction service. "
        "The model files are loaded by that service, not by Streamlit."
    )
    st.caption(f"Prediction API: {API_URL}")
    try:
        health = call_api("/health")
        if health.get("ready"):
            st.success("Prediction API is ready")
        else:
            st.warning("API is running, but models are unavailable")
    except (HTTPError, URLError, TimeoutError):
        st.error("Prediction API is not reachable")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<p class="sub-header">📝 Enter Ticket Details</p>', unsafe_allow_html=True)
    with st.form(key="ticket_form"):
        ticket_subject = st.text_input("Ticket Subject", placeholder="Brief summary of the issue")
        ticket_description = st.text_area("Ticket Description", placeholder="Detailed description of the issue...", height=150)
        input_col1, input_col2 = st.columns(2)
        with input_col1:
            customer_gender = st.selectbox("Customer Gender", ["Male", "Female", "Others"])
            customer_age = st.number_input("Customer Age", min_value=1, max_value=120, value=30)
        with input_col2:
            product_purchased = st.text_input("Product Purchased", placeholder="Product Name")
            purchase_date = st.date_input("Date of Purchase", value=datetime.now().date())
        ticket_channel = st.selectbox("Ticket Channel", ["Social media", "Chat", "Email", "Phone"])
        submitted = st.form_submit_button("🔮 Predict Ticket Metrics", use_container_width=True)

with col2:
    st.markdown('<p class="sub-header">📊 Predictions</p>', unsafe_allow_html=True)
    if submitted:
        if not ticket_description.strip():
            st.warning("Enter a ticket description before predicting.")
        else:
            payload = {
                "subject": ticket_subject,
                "description": ticket_description,
                "customer_gender": customer_gender,
                "customer_age": customer_age,
                "product_purchased": product_purchased or "Unknown",
                "purchase_date": purchase_date.isoformat(),
                "ticket_channel": ticket_channel,
                "satisfaction_rating": 4,
            }
            try:
                with st.spinner("Analyzing ticket..."):
                    prediction = call_api("/v1/predict", payload)
                display_prediction(prediction)
            except HTTPError as exc:
                try:
                    detail = json.loads(exc.read().decode("utf-8")).get("detail", str(exc))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    detail = str(exc)
                st.error(f"Prediction API error: {detail}")
            except (URLError, TimeoutError) as exc:
                st.error(f"Could not reach the prediction API: {exc}")
    else:
        st.info("Enter ticket details and click Predict.")