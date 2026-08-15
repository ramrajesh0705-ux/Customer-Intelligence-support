import streamlit as st

st.set_page_config(page_title="Customer Support Intelligence", page_icon="📊")

st.title("🏠 Customer Support Intelligence Dashboard")

st.markdown("""
## Project Overview

Welcome to the **Customer Support Intelligence** system! This AI-powered platform helps analyze and predict customer support ticket outcomes to improve response times and customer satisfaction.

### Key Features

🔍 **Exploratory Data Analysis (EDA)**: Dive deep into ticket data with interactive visualizations and insights.

🎯 **Predictive Analytics**: Predict ticket type, priority, and resolution time using machine learning models.

📈 **Performance Monitoring**: Track key metrics and model performance over time.

### How It Works

1. **Data Collection**: Customer support tickets are collected with details like subject, description, channel, and customer information.

2. **Preprocessing**: Text data is cleaned and transformed, categorical variables are encoded.

3. **Model Training**: Multiple ML models are trained for different prediction tasks:
   - Ticket Type Classification
   - Priority Prediction
   - Resolution Time Estimation

4. **Inference**: Real-time predictions for new tickets to optimize support workflows.

### Technologies Used

- **Machine Learning**: LightGBM, Scikit-learn, Transformers
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Web Framework**: Streamlit
- **Experiment Tracking**: MLflow

Navigate to the sidebar to explore the EDA dashboard or make predictions!
""")

# Add some statistics or quick overview
st.subheader("Quick Stats")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Models Deployed", "3", "Active")

with col2:
    st.metric("Prediction Types", "3", "Available")

with col3:
    st.metric("Data Points", "1000+", "Processed")
