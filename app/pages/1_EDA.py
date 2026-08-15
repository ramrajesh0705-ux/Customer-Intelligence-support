# app/pages/1_EDA.py
import sys
import os
from pathlib import Path

# Add the project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="EDA - Customer Support Intelligence",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .eda-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .eda-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .eda-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    .stat-box {
        background-color: white;
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s;
    }
    .stat-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .stat-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f77b4;
    }
    .stat-label {
        color: #666;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    .insight-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .insight-box h4 {
        margin-top: 0;
        color: #333;
    }
    .insight-box ul {
        margin-bottom: 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="eda-header">
    <h1>📊 Exploratory Data Analysis</h1>
    <p>Comprehensive analysis of ticket distributions, text patterns, and class imbalances</p>
</div>
""", unsafe_allow_html=True)

# ============= DATA LOADING =============
@st.cache_data
def load_data():
    """Load and prepare data for EDA"""
    try:
        possible_paths = [
            Path(__file__).resolve().parent.parent.parent / 'data' / 'sample.csv',
            Path(__file__).resolve().parent.parent.parent / 'data' / 'sample_tickets.csv',
            Path(__file__).resolve().parent.parent / 'data' / 'sample.csv',
            Path('./data/sample.csv'),
            Path('../data/sample.csv')
        ]
        
        for path in possible_paths:
            if path.exists():
                df = pd.read_csv(path)
                st.success(f"✅ Data loaded from: {path.name} ({len(df)} rows)")
                break
        else:
            st.info("ℹ️ No data file found. Generating sample data for demonstration...")
            df = create_sample_data()
        
        # Standardize column names
        column_mapping = {
            'Ticket Type': 'ticket_type',
            'TicketType': 'ticket_type',
            'type': 'ticket_type',
            'Ticket Priority': 'ticket_priority',
            'Priority': 'ticket_priority',
            'Channel': 'ticket_channel',
            'Ticket Channel': 'ticket_channel',
            'Satisfaction': 'customer_satisfaction_rating',
            'Rating': 'customer_satisfaction_rating',
            'Time to Resolution': 'time_to_resolution',
            'Ticket Subject': 'ticket_subject',
            'Ticket Description': 'ticket_description',
            'Ticket ID': 'ticket_id'
        }
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required_cols = ['ticket_type', 'ticket_priority', 'ticket_channel', 'customer_satisfaction_rating']
        for col in required_cols:
            if col not in df.columns:
                if col == 'ticket_type':
                    df['ticket_type'] = 'General Inquiry'
                elif col == 'ticket_priority':
                    df['ticket_priority'] = 'Medium'
                elif col == 'ticket_channel':
                    df['ticket_channel'] = 'Email'
                elif col == 'customer_satisfaction_rating':
                    df['customer_satisfaction_rating'] = np.random.randint(1, 6, len(df))
        
        # Create ticket text if needed
        if 'ticket_text' not in df.columns:
            if 'ticket_subject' in df.columns and 'ticket_description' in df.columns:
                df['ticket_text'] = df['ticket_subject'].fillna('') + ' ' + df['ticket_description'].fillna('')
            else:
                df['ticket_text'] = df['ticket_type'] + ' issue reported'
        
        # Create time_to_resolution if needed or convert to numeric
        if 'time_to_resolution' not in df.columns:
            df['time_to_resolution'] = np.random.exponential(48, len(df)).clip(1, 168).round(1)
        else:
            # Convert to numeric, coercing errors
            df['time_to_resolution'] = pd.to_numeric(df['time_to_resolution'], errors='coerce')
            # Fill NaN values with random values
            if df['time_to_resolution'].isna().any():
                mask = df['time_to_resolution'].isna()
                df.loc[mask, 'time_to_resolution'] = np.random.exponential(48, mask.sum()).clip(1, 168).round(1)
        
        # Create ticket_id if needed
        if 'ticket_id' not in df.columns:
            df['ticket_id'] = [f'TKT-{i:04d}' for i in range(len(df))]
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return create_sample_data()

def create_sample_data():
    """Create sample ticket data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    ticket_types = ['Technical Issue', 'Billing Question', 'Feature Request', 'Account Management', 'General Inquiry']
    priorities = ['Low', 'Medium', 'High', 'Critical']
    channels = ['Email', 'Phone', 'Chat', 'Social Media', 'In-Person']
    
    df = pd.DataFrame({
        'ticket_id': [f'TKT-{i:04d}' for i in range(n_samples)],
        'ticket_type': np.random.choice(ticket_types, n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'ticket_priority': np.random.choice(priorities, n_samples, p=[0.2, 0.35, 0.3, 0.15]),
        'ticket_channel': np.random.choice(channels, n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'time_to_resolution': np.random.exponential(48, n_samples).clip(1, 168).round(1),
        'customer_satisfaction_rating': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.1, 0.2, 0.35, 0.3]),
        'ticket_subject': [f'Issue with {np.random.choice(["login", "payment", "dashboard", "account", "feature"])}' for _ in range(n_samples)],
        'ticket_description': [f'Customer reported issue with {np.random.choice(["login", "payment", "dashboard", "account", "feature"])}' for _ in range(n_samples)]
    })
    
    df['ticket_text'] = df['ticket_subject'] + ' ' + df['ticket_description']
    return df

# ============= LOAD DATA =============
df = load_data()

# ============= SIDEBAR FILTERS =============
with st.sidebar:
    st.header("🔍 Filters")
    
    selected_type = st.selectbox("Ticket Type", ['All'] + sorted(df['ticket_type'].unique().tolist()))
    selected_priority = st.selectbox("Priority", ['All'] + sorted(df['ticket_priority'].unique().tolist()))
    selected_channel = st.selectbox("Channel", ['All'] + sorted(df['ticket_channel'].unique().tolist()))
    selected_rating = st.selectbox("Satisfaction Rating", ['All'] + sorted(df['customer_satisfaction_rating'].unique().tolist()))
    
    # Apply filters
    filtered_df = df.copy()
    if selected_type != 'All':
        filtered_df = filtered_df[filtered_df['ticket_type'] == selected_type]
    if selected_priority != 'All':
        filtered_df = filtered_df[filtered_df['ticket_priority'] == selected_priority]
    if selected_channel != 'All':
        filtered_df = filtered_df[filtered_df['ticket_channel'] == selected_channel]
    if selected_rating != 'All':
        filtered_df = filtered_df[filtered_df['customer_satisfaction_rating'] == selected_rating]
    
    st.divider()
    st.info(f"📊 Showing {len(filtered_df):,} of {len(df):,} tickets")

# ============= OVERVIEW METRICS =============
st.markdown("### 📊 Dataset Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-value">{len(filtered_df):,}</div>
        <div class="stat-label">Total Tickets</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-value">{filtered_df['ticket_type'].nunique()}</div>
        <div class="stat-label">Ticket Types</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-value">{filtered_df['ticket_priority'].nunique()}</div>
        <div class="stat-label">Priority Levels</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-value">{filtered_df['ticket_channel'].nunique()}</div>
        <div class="stat-label">Channels</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    avg_satisfaction = filtered_df['customer_satisfaction_rating'].mean()
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-value">{avg_satisfaction:.1f}</div>
        <div class="stat-label">Avg Satisfaction</div>
    </div>
    """, unsafe_allow_html=True)

# ============= TABS =============
tab1, tab2, tab3 = st.tabs([
    "📈 Distributions",
    "📝 Text Analysis",
    "⚖️ Class Imbalance & Insights"
])

# ============= TAB 1: DISTRIBUTIONS =============
with tab1:
    st.subheader("📈 Distribution Analysis")
    
    # Row 1: Ticket Type and Priority
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Ticket Type Distribution")
        type_counts = filtered_df['ticket_type'].value_counts()
        
        fig = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="Ticket Types Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Ticket Priority Distribution")
        priority_counts = filtered_df['ticket_priority'].value_counts()
        
        colors = {'Low': '#2ca02c', 'Medium': '#ff7f0e', 'High': '#d62728', 'Critical': '#8c564b'}
        fig = px.bar(
            x=priority_counts.index,
            y=priority_counts.values,
            title="Priority Distribution",
            color=priority_counts.index,
            color_discrete_map=colors
        )
        fig.update_layout(xaxis_title="Priority", yaxis_title="Count", height=400)
        fig.update_traces(text=priority_counts.values, textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Row 2: Channel and Satisfaction
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### Ticket Channel Distribution")
        channel_counts = filtered_df['ticket_channel'].value_counts()
        
        fig = px.pie(
            values=channel_counts.values,
            names=channel_counts.index,
            title="Channel Distribution",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            hole=0.3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        st.markdown("#### Customer Satisfaction Rating")
        rating_counts = filtered_df['customer_satisfaction_rating'].value_counts().sort_index()
        
        fig = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            title="Satisfaction Rating Distribution",
            color=rating_counts.index,
            color_discrete_sequence=px.colors.sequential.RdBu_r
        )
        fig.update_layout(xaxis_title="Rating", yaxis_title="Count", height=400)
        fig.update_traces(text=rating_counts.values, textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

# ============= TAB 2: TEXT ANALYSIS =============
with tab2:
    st.subheader("📝 Text Analysis")
    
    # Text length and word count distributions
    filtered_df['text_length'] = filtered_df['ticket_text'].astype(str).str.len()
    filtered_df['word_count'] = filtered_df['ticket_text'].astype(str).str.split().str.len()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Text Length Distribution")
        fig = px.histogram(
            filtered_df,
            x='text_length',
            title="Ticket Text Length Distribution",
            nbins=50,
            color_discrete_sequence=['#2ca02c'],
            marginal='box'
        )
        fig.update_layout(xaxis_title="Text Length", yaxis_title="Count", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Word Count Distribution")
        fig = px.box(
            filtered_df,
            y='word_count',
            title="Word Count Distribution",
            color_discrete_sequence=['#ff7f0e']
        )
        fig.update_layout(yaxis_title="Word Count", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Text length by ticket type
    st.markdown("#### Text Length by Ticket Type")
    
    fig = px.box(
        filtered_df,
        x='ticket_type',
        y='text_length',
        title="Text Length Distribution by Ticket Type",
        color='ticket_type',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_layout(xaxis_title="Ticket Type", yaxis_title="Text Length", height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Word Cloud
    st.markdown("#### Word Cloud - Ticket Descriptions")
    
    try:
        text = ' '.join(filtered_df['ticket_text'].dropna().astype(str))
        if len(text) > 0:
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=100,
                colormap='viridis'
            ).generate(text)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Word Cloud - Ticket Descriptions', fontsize=14, fontweight='bold')
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Not enough text data to generate word cloud")
    except Exception as e:
        st.warning(f"Could not generate word cloud: {e}")

# ============= TAB 3: CLASS IMBALANCE & INSIGHTS =============
with tab3:
    st.subheader("⚖️ Class Imbalance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Ticket Type Distribution")
        type_counts = filtered_df['ticket_type'].value_counts()
        
        fig = px.bar(
            x=type_counts.index,
            y=type_counts.values,
            title="Ticket Type Class Distribution",
            color=type_counts.index,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(xaxis_title="Ticket Type", yaxis_title="Count", height=400)
        fig.update_traces(text=type_counts.values, textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Imbalance metrics
        total = sum(type_counts.values)
        st.markdown(f"""
        <div class="insight-box">
            <h4>📊 Type Imbalance</h4>
            <ul>
                <li><b>Most Common:</b> {type_counts.index[0]} ({type_counts.values[0]/total:.1%})</li>
                <li><b>Least Common:</b> {type_counts.index[-1]} ({type_counts.values[-1]/total:.1%})</li>
                <li><b>Ratio:</b> {type_counts.values[0]/type_counts.values[-1]:.1f}x</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Ticket Priority Distribution")
        priority_counts = filtered_df['ticket_priority'].value_counts()
        
        colors = {'Low': '#2ca02c', 'Medium': '#ff7f0e', 'High': '#d62728', 'Critical': '#8c564b'}
        fig = px.bar(
            x=priority_counts.index,
            y=priority_counts.values,
            title="Priority Class Distribution",
            color=priority_counts.index,
            color_discrete_map=colors
        )
        fig.update_layout(xaxis_title="Priority", yaxis_title="Count", height=400)
        fig.update_traces(text=priority_counts.values, textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        total_priority = sum(priority_counts.values)
        st.markdown(f"""
        <div class="insight-box">
            <h4>📊 Priority Imbalance</h4>
            <ul>
                <li><b>Most Common:</b> {priority_counts.index[0]} ({priority_counts.values[0]/total_priority:.1%})</li>
                <li><b>Least Common:</b> {priority_counts.index[-1]} ({priority_counts.values[-1]/total_priority:.1%})</li>
                <li><b>Ratio:</b> {priority_counts.values[0]/priority_counts.values[-1]:.1f}x</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Cross-tab heatmap
    st.markdown("#### Cross-Tab Analysis: Type vs Priority")
    cross_tab = pd.crosstab(filtered_df['ticket_type'], filtered_df['ticket_priority'])
    
    fig = px.imshow(
        cross_tab,
        text_auto=True,
        color_continuous_scale='Blues',
        title="Heatmap: Ticket Type vs Priority"
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)
    
    # Priority vs Resolution Time (if available)
    if 'time_to_resolution' in filtered_df.columns:
        st.markdown("#### Priority vs Resolution Time")
        
        fig = px.box(
            filtered_df,
            x='ticket_priority',
            y='time_to_resolution',
            title="Resolution Time Distribution by Priority",
            color='ticket_priority',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(xaxis_title="Priority", yaxis_title="Resolution Time (Hours)", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Channel Performance - FIXED: Only aggregate numeric columns
    if 'ticket_channel' in filtered_df.columns:
        st.markdown("#### Channel Performance Metrics")
        
        # Select only numeric columns for aggregation
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            # Create aggregation dict for numeric columns only
            agg_dict = {col: 'mean' for col in numeric_cols}
            # Add count for ticket_id if it exists
            if 'ticket_id' in filtered_df.columns:
                agg_dict['ticket_id'] = 'count'
            elif len(filtered_df.columns) > 0:
                # Use first column as count if ticket_id doesn't exist
                agg_dict[filtered_df.columns[0]] = 'count'
            
            channel_perf = filtered_df.groupby('ticket_channel').agg(agg_dict).round(2)
            
            # Rename columns for better display
            rename_map = {
                'time_to_resolution': 'Avg Resolution (hrs)',
                'customer_satisfaction_rating': 'Avg Satisfaction',
                'ticket_id': 'Ticket Count'
            }
            channel_perf = channel_perf.rename(columns=rename_map)
            
            st.dataframe(channel_perf, use_container_width=True)
            
            # Visualize channel performance if we have both resolution and satisfaction
            if 'time_to_resolution' in channel_perf.columns and 'customer_satisfaction_rating' in channel_perf.columns:
                fig = px.scatter(
                    channel_perf.reset_index(),
                    x='Avg Resolution (hrs)',
                    y='Avg Satisfaction',
                    size='Ticket Count' if 'Ticket Count' in channel_perf.columns else None,
                    hover_name='ticket_channel',
                    title="Channel Performance: Resolution Time vs Satisfaction",
                    color='ticket_channel',
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns available for performance metrics")

# ============= KEY INSIGHTS SUMMARY =============
st.divider()

# Safely calculate metrics with error handling
def safe_mean(series):
    try:
        return series.mean()
    except:
        return 0

def safe_mode(series):
    try:
        return series.mode().iloc[0] if not series.empty and not series.mode().empty else 'N/A'
    except:
        return 'N/A'

# Calculate metrics safely
total_tickets = len(filtered_df)
most_common_type = safe_mode(filtered_df['ticket_type'])
most_common_type_pct = (filtered_df['ticket_type'].value_counts().iloc[0] / total_tickets * 100) if not filtered_df['ticket_type'].empty else 0
most_common_priority = safe_mode(filtered_df['ticket_priority'])
most_common_priority_pct = (filtered_df['ticket_priority'].value_counts().iloc[0] / total_tickets * 100) if not filtered_df['ticket_priority'].empty else 0
avg_resolution = safe_mean(filtered_df['time_to_resolution']) if 'time_to_resolution' in filtered_df.columns else 0
avg_satisfaction = safe_mean(filtered_df['customer_satisfaction_rating']) if 'customer_satisfaction_rating' in filtered_df.columns else 0
most_common_channel = safe_mode(filtered_df['ticket_channel']) if 'ticket_channel' in filtered_df.columns else 'N/A'
most_common_channel_pct = (filtered_df['ticket_channel'].value_counts().iloc[0] / total_tickets * 100) if 'ticket_channel' in filtered_df.columns and not filtered_df['ticket_channel'].empty else 0

st.markdown(f"""
<div class="insight-box">
    <h4>💡 Key Insights Summary</h4>
    <ul>
        <li>📊 <b>Total Tickets Analyzed:</b> {total_tickets:,}</li>
        <li>🎯 <b>Most Common Ticket Type:</b> {most_common_type} ({most_common_type_pct:.1f}%)</li>
        <li>⚡ <b>Most Frequent Priority Level:</b> {most_common_priority} ({most_common_priority_pct:.1f}%)</li>
        <li>📈 <b>Average Resolution Time:</b> {avg_resolution:.1f} hours</li>
        <li>⭐ <b>Average Satisfaction Rating:</b> {avg_satisfaction:.1f}/5.0</li>
        <li>📢 <b>Most Used Channel:</b> {most_common_channel} ({most_common_channel_pct:.1f}%)</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9rem;'>
    📊 EDA Dashboard | Powered by Plotly & Matplotlib
</div>
""", unsafe_allow_html=True)