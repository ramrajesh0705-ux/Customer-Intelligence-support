# prediction.py
# -*- coding: utf-8 -*-
"""
Customer Intelligence Ticket Predictor
A Streamlit app for predicting ticket category, priority, and resolution time
"""

import streamlit as st
import pandas as pd
import torch
import joblib
import numpy as np
import warnings
import pickle
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")

# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Ticket Intelligence Predictor",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        font-weight: 500;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .prediction-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
    }
    .stButton button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
    }
    .stButton button:hover {
        background-color: #145a8d;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# Load models with caching
# =========================================================
@st.cache_resource
def load_models():
    """Load all models and tokenizers with caching"""
    
    # Define paths - UPDATE THESE PATHS FOR YOUR SYSTEM
    model_path = '/content/drive/MyDrive/CustomerIntelligence/best_model/checkpoint-3815'
    label_encoder_path = '/content/drive/MyDrive/CustomerIntelligence/best_model/checkpoint-3815/label_encoder.pkl'
    resolution_time_model_path = '/content/drive/MyDrive/CustomerIntelligence/mlruns/718547084090739768/models/m-3a8e48a1b4d0419095dfb09c2c63593d/artifacts/model.pkl'
    priority_classifier_path = '/content/drive/MyDrive/CustomerIntelligence/mlruns/158184397920343216/models/m-4462f497539a49e4b3bef147035c62f4/artifacts/model.skops'
    
    models = {
        'tokenizer': None,
        'category_model': None,
        'label_encoder': None,
        'resolution_time_predictor': None,
        'priority_model': None,
        'priority_label_encoder': None
    }
    
    # Try to import transformers only if needed
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            models['tokenizer'] = tokenizer
            st.success("✅ Tokenizer loaded")
        except Exception as e:
            try:
                parent_model_path = '/content/drive/MyDrive/CustomerIntelligence/best_model'
                tokenizer = AutoTokenizer.from_pretrained(parent_model_path)
                models['tokenizer'] = tokenizer
                st.success("✅ Tokenizer loaded from parent")
            except Exception as e2:
                st.warning(f"⚠️ Tokenizer not loaded: {e2}")
        
        # Load category classifier model
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.eval()
            models['category_model'] = model
            st.success("✅ Category classifier loaded")
        except Exception as e:
            st.warning(f"⚠️ Category classifier not loaded: {e}")
            
    except ImportError:
        st.warning("⚠️ Transformers library not available. Category prediction disabled.")
    
    # Load label encoder
    try:
        label_encoder = joblib.load(label_encoder_path)
        models['label_encoder'] = label_encoder
        st.success("✅ Label encoder loaded")
    except Exception as e:
        st.warning(f"⚠️ Label encoder not loaded: {e}")
    
    # Load resolution time model
    try:
        resolution_time_predictor = joblib.load(resolution_time_model_path)
        if hasattr(resolution_time_predictor, 'predict'):
            models['resolution_time_predictor'] = resolution_time_predictor
            st.success("✅ Resolution time model loaded")
        else:
            st.warning("⚠️ Resolution time model invalid")
    except Exception as e:
        st.warning(f"⚠️ Resolution time model not loaded: {e}")
    
    # Load priority classifier
    try:
        import skops.io as sio
        priority_model = sio.load(priority_classifier_path, trusted=[
            'sklearn', 'numpy', 'lightgbm', 'collections',
            'collections.OrderedDict', 'lightgbm.basic.Booster',
            'lightgbm.sklearn.LGBMClassifier',
            'sklearn.compose._column_transformer._RemainderColsList'
        ])
        if hasattr(priority_model, 'predict'):
            models['priority_model'] = priority_model
            st.success("✅ Priority classifier loaded")
        else:
            st.warning("⚠️ Priority model invalid")
    except Exception as e:
        st.warning(f"⚠️ Priority classifier not loaded: {e}")
    
    # Create priority label encoder
    try:
        from sklearn.preprocessing import LabelEncoder
        priority_label_encoder = LabelEncoder()
        priority_label_encoder.fit(['Low', 'Medium', 'High', 'Critical'])
        models['priority_label_encoder'] = priority_label_encoder
    except:
        pass
    
    return models

# =========================================================
# Prediction functions
# =========================================================
def classify_ticket_category(text, tokenizer, model, label_encoder):
    """Classify ticket category"""
    if model is None or tokenizer is None or label_encoder is None:
        return "Model not available", 0.0
    
    try:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, 
                          padding=True, max_length=512)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_class_id = torch.argmax(probabilities, dim=1).item()
        predicted_label = label_encoder.inverse_transform([predicted_class_id])[0]
        confidence = probabilities[0][predicted_class_id].item()
        
        return predicted_label, confidence
    except Exception as e:
        return f"Error: {e}", 0.0

def extract_features_for_prediction(input_data_df, model_type, results):
    """
    Extract features for prediction models matching training column names
    
    Training columns used:
    - Customer Gender (categorical)
    - Customer Age (numerical)
    - Product Purchased (categorical)
    - Ticket Type (categorical)
    - Ticket Priority (categorical)
    - Ticket Channel (categorical)
    - Customer Satisfaction Rating (numerical)
    - Ticket Text (text feature)
    """
    df_processed = input_data_df.copy()
    
    # Ensure all required columns exist with correct names
    if model_type == 'ticketpriority':
        try:
            # Convert dates
            if 'Date of Purchase' in df_processed.columns:
                df_processed["Date of Purchase"] = pd.to_datetime(df_processed["Date of Purchase"], errors='coerce')
            if 'First Response Time' in df_processed.columns:
                df_processed['First Response Time'] = pd.to_datetime(df_processed['First Response Time'], errors='coerce')
            
            # Create Days Since Purchase
            if 'Date of Purchase' in df_processed.columns:
                reference_date = df_processed['First Response Time'].max() if 'First Response Time' in df_processed.columns else pd.Timestamp.now()
                df_processed["Days Since Purchase"] = (reference_date - df_processed["Date of Purchase"]).dt.days
                df_processed["Days Since Purchase"] = df_processed["Days Since Purchase"].fillna(0)
            
            # Create Ticket Text
            df_processed["Ticket Text"] = (
                df_processed.get("Ticket Subject", "").fillna("") + " " +
                df_processed.get("Ticket Description", "").fillna("")
            )
            
            # Drop unnecessary columns
            drop_columns = ["Ticket ID", "Customer Name", "Customer Email", "Date of Purchase", 
                          "First Response Time", "Ticket Subject", "Ticket Description",
                          "Ticket Status", "Resolution", "Time to Resolution", "Ticket Priority"]
            drop_columns = [col for col in drop_columns if col in df_processed.columns]
            df_processed = df_processed.drop(columns=drop_columns, errors='ignore')
            
            # Ensure all required columns exist
            required_cols = ['Days Since Purchase', 'Ticket Text']
            for col in required_cols:
                if col not in df_processed.columns:
                    df_processed[col] = 0 if col == 'Days Since Purchase' else ''
            
            return df_processed
        except Exception as e:
            st.error(f"Error in priority features: {e}")
            return df_processed
    
    else:  
        # resolutiontime
         try:
            # === FIX: Use the same column names as training ===
            # 1. Ensure Customer Gender exists
            if 'Customer Gender' not in df_processed.columns:
                df_processed['Customer Gender'] = 'Unknown'
            
            # 2. Ensure Customer Age exists (numerical)
            if 'Customer Age' not in df_processed.columns:
                df_processed['Customer Age'] = 30
            
            # 3. Ensure Product Purchased exists
            if 'Product Purchased' not in df_processed.columns:
                df_processed['Product Purchased'] = 'Unknown'
            
            # 4. Ensure Ticket Type exists (use predicted category)
            if 'Ticket Type' not in df_processed.columns:
                df_processed['Ticket Type'] = results.get('category', 'General Inquiry')
            
            # 5. Ensure Ticket Priority exists (use predicted priority)
            if 'Ticket Priority' not in df_processed.columns:
                df_processed['Ticket Priority'] = results.get('priority', 'Medium')
            
            # 6. Ensure Ticket Channel exists
            if 'Ticket Channel' not in df_processed.columns:
                df_processed['Ticket Channel'] = 'Email'
            
            # 7. Ensure Customer Satisfaction Rating exists (numerical)
            if 'Customer Satisfaction Rating' not in df_processed.columns:
                df_processed['Customer Satisfaction Rating'] = 4
            
            # 8. Create Ticket Text
            df_processed["Ticket Text"] = (
                df_processed.get("Ticket Subject", "").fillna("") + " " +
                df_processed.get("Ticket Description", "").fillna("")
            )
            
            # 9. Select only the columns used in training
            required_columns = [
                'Customer Gender',
                'Customer Age', 
                'Product Purchased',
                'Ticket Type',
                'Ticket Priority',
                'Ticket Channel',
                'Customer Satisfaction Rating',
                'Ticket Text'
            ]
            
            # Ensure all required columns exist
            for col in required_columns:
                if col not in df_processed.columns:
                    if col == 'Customer Age':
                        df_processed[col] = 30
                    elif col == 'Customer Satisfaction Rating':
                        df_processed[col] = 4
                    elif col in ['Customer Gender', 'Product Purchased', 'Ticket Type', 
                                'Ticket Priority', 'Ticket Channel']:
                        df_processed[col] = 'Unknown'
                    elif col == 'Ticket Text':
                        df_processed[col] = ''
            
            # Keep only required columns
            df_processed = df_processed[required_columns]
            
            # Convert numeric columns to correct types
            if 'Customer Age' in df_processed.columns:
                df_processed['Customer Age'] = pd.to_numeric(df_processed['Customer Age'], errors='coerce').fillna(30)
            if 'Customer Satisfaction Rating' in df_processed.columns:
                df_processed['Customer Satisfaction Rating'] = pd.to_numeric(
                    df_processed['Customer Satisfaction Rating'], errors='coerce'
                ).fillna(4)
            
            return df_processed
            
         except Exception as e:
               # Raise the exception with full traceback to be caught upstream
             raise RuntimeError(f"Feature extraction failed: {e}\n{traceback.format_exc()}")

def predict_ticket_metrics(input_df, models):
    """Predict all ticket metrics"""
    results = {
        'category': None,
        'category_confidence': 0.0,
        'priority': None,
        'resolution_time': None
    }
    
    # Category prediction
    if models.get('tokenizer') and models.get('category_model') and models.get('label_encoder'):
        if 'Ticket Description' in input_df.columns and 'Ticket Subject' in input_df.columns:
            text = str(input_df['Ticket Subject'].fillna('').iloc[0]) + ' ' + str(input_df['Ticket Description'].fillna('').iloc[0])
            category, confidence = classify_ticket_category(
                text,
                models.get('tokenizer'),
                models.get('category_model'),
                models.get('label_encoder')
            )
            results['category'] = category
            results['category_confidence'] = confidence
    
    # Priority prediction
    if models.get('priority_model'):
        try:
            priority_features = extract_features_for_prediction(input_df.copy(), 'ticketpriority', results)
            pred = models['priority_model'].predict(priority_features)[0]
            
            if models.get('priority_label_encoder'):
                try:
                    predicted_priority = models['priority_label_encoder'].inverse_transform([pred])[0]
                except:
                    predicted_priority = str(pred)
            else:
                predicted_priority = str(pred)
            results['priority'] = predicted_priority
        except Exception as e:
            results['priority'] = f"Error: {e}"
    
    # Resolution time prediction
    if models.get('resolution_time_predictor'):
        try:
            # Prepare features
            resolution_features = extract_features_for_prediction(input_df.copy(), 'resolutiontime', results)
            
            # --- DEBUGGING OUTPUT ---
            debug_info = {
                'features_shape': resolution_features.shape,
                'features_columns': resolution_features.columns.tolist(),
                'first_row': resolution_features.iloc[0].to_dict() if len(resolution_features) > 0 else None,
                'data_types': resolution_features.dtypes.to_dict()
            }
            st.session_state['debug_resolution'] = debug_info  # Store for display later
            
            # Try prediction
            predicted_resolution_time = models['resolution_time_predictor'].predict(resolution_features)[0]
            results['resolution_time'] = float(predicted_resolution_time)
            
        except Exception as e:
            # Capture full traceback and store in session state for display
            error_trace = traceback.format_exc()
            st.session_state['debug_resolution_error'] = error_trace
            st.session_state['debug_resolution_exception'] = str(e)
            results['resolution_time'] = None
            # Re-raise to show error in the UI
            raise RuntimeError(f"Resolution time prediction failed: {e}\n{error_trace}")
    
    return results

# =========================================================
# Main app
# =========================================================
def main():
    # Header
    st.markdown('<p class="main-header">🎫 Ticket Intelligence Predictor</p>', unsafe_allow_html=True)
    st.markdown("Predict ticket category, priority, and estimated resolution time using AI")
    st.divider()
    
    # Load models
    with st.spinner("Loading AI models... This may take a moment."):
        models = load_models()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 About")
        st.info(
            "This app uses AI models to analyze support tickets:\n\n"
            "• **Category Classifier** - Identifies ticket type\n"
            "• **Priority Predictor** - Determines urgency level\n"
            "• **Resolution Time Predictor** - Estimates resolution time\n\n"
            "Enter ticket details below to get predictions."
        )
        st.markdown("---")
        st.caption("Built with Streamlit • Customer Intelligence")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<p class="sub-header">📝 Enter Ticket Details</p>', unsafe_allow_html=True)
        
        # Create form for better UX
        with st.form(key="ticket_form"):
            ticket_subject = st.text_input(
                "Ticket Subject",
                placeholder="Brief summary of the issue",
                help="Enter the main subject of the ticket"
            )
            
            ticket_description = st.text_area(
                "Ticket Description",
                placeholder="Detailed description of the issue...",
                height=150,
                help="Provide a detailed description of the ticket"
            )
            col1a, col1b = st.columns(2)
            with col1a:
                customer_gender = st.selectbox(
                    "Customer Gender",
                    ["Male", "Female", "Others"]
                )
            
            with col1b:
                customer_age = st.number_input(
                    "Customer Age",
                    min_value=1,
                    max_value=120,
                    value=30,
                    help="Enter the customer's age"
                )
            
            col1c, col1d, col1e = st.columns(3)
            with col1c:
                product_purchased = st.text_input(
                    "Product Purchased",
                    placeholder="Product Name",
                    help="Enter the product name"
                )
            
            with col1d:
                purchase_date = st.date_input(
                    "Date of Purchase",
                    value=datetime.now().date()
                )
            with col1e:
                ticket_channel = st.selectbox(
                    "Ticket Channel",
                    ['Social media', 'Chat', 'Email', 'Phone']
                )
            
            # Customer Satisfaction Rating
            satisfaction_rating = 4
            
            # Submit button inside form
            submitted = st.form_submit_button("🔮 Predict Ticket Metrics", use_container_width=True)
    
    # Prediction results
    with col2:
        st.markdown('<p class="sub-header">📊 Predictions</p>', unsafe_allow_html=True)
        
        if submitted and ticket_description.strip():
            # Prepare input DataFrame with ALL required columns
            input_data = pd.DataFrame([{
                'Ticket ID': '',
                'Customer Name': '',
                'Customer Email': '',
                'Customer Age': customer_age,
                'Customer Gender': customer_gender,
                'Product Purchased': product_purchased,
                'Date of Purchase': purchase_date,
                'Ticket Type': '',
                'Ticket Subject': ticket_subject,
                'Ticket Description': ticket_description,
                'Ticket Status': '',
                'Resolution': '',
                'Ticket Priority': '',
                'Ticket Channel': ticket_channel,
                'First Response Time': datetime.now().date(),
                'Time to Resolution': '',
                'Customer Satisfaction Rating': satisfaction_rating
            }])
            
            with st.spinner("Analyzing ticket..."):
                predictions = predict_ticket_metrics(input_data, models)
            
            # Display predictions
            st.markdown("#### 🏷️ Category")
            if predictions['category'] and "Error" not in str(predictions['category']):
                confidence = predictions['category_confidence']
                st.markdown(f"""
                    <div class="prediction-card">
                        <strong>{predictions['category']}</strong><br>
                        <small>Confidence: {confidence:.1%}</small>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("ℹ️ Category prediction not available")
            
            st.markdown("#### ⚡ Priority")
            if predictions['priority'] and "Error" not in str(predictions['priority']):
                priority_colors = {
                    'Critical': '🔴',
                    'High': '🟠',
                    'Medium': '🟡',
                    'Low': '🟢'
                }
                emoji = priority_colors.get(predictions['priority'], '⚪')
                st.markdown(f"""
                    <div class="prediction-card" style="border-left-color: {'#e74c3c' if predictions['priority'] in ['Critical', 'High'] else '#f39c12' if predictions['priority'] == 'Medium' else '#2ecc71'};">
                        <strong>{emoji} {predictions['priority']}</strong>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("ℹ️ Priority prediction not available")
            
            st.markdown("#### ⏱️ Resolution Time")
            if predictions['resolution_time'] and isinstance(predictions['resolution_time'], (int, float)):
                hours = predictions['resolution_time']
                if hours < 24:
                    time_str = f"{hours:.1f} hours"
                else:
                    days = hours / 24
                    time_str = f"{days:.1f} days ({hours:.1f} hours)"
                
                st.markdown(f"""
                    <div class="prediction-card" style="border-left-color: #3498db;">
                        <strong>{time_str}</strong>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("ℹ️ Resolution time prediction not available")
            
        else:
            st.info("👆 Enter ticket details and click 'Predict'")
    
    # Model status
    st.divider()
    with st.expander("ℹ️ Model Status", expanded=False):
        col_status1, col_status2, col_status3 = st.columns(3)
        with col_status1:
            st.markdown("**Category Classifier**")
            if models.get('category_model'):
                st.success("✅ Loaded")
            else:
                st.warning("⚠️ Not Available")
        
        with col_status2:
            st.markdown("**Priority Predictor**")
            if models.get('priority_model'):
                st.success("✅ Loaded")
            else:
                st.warning("⚠️ Not Available")
        
        with col_status3:
            st.markdown("**Resolution Time Predictor**")
            if models.get('resolution_time_predictor'):
                st.success("✅ Loaded")
            else:
                st.warning("⚠️ Not Available")

if __name__ == "__main__":
    main()