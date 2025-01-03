import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import sys
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Enhanced Alarm Analysis", 
    page_icon="🚨", 
    layout="wide"
)

def categorize_alarm(message):
    """Categorize alarms into main groups"""
    msg = message.lower()
    if 'water pump flow' in msg:
        subcategory = 'Flow Rate'
        if 'high' in msg or 'above' in msg:
            detail = 'High Flow'
        elif 'low' in msg or 'below' in msg:
            detail = 'Low Flow'
        else:
            detail = 'Flow Issue'
        return 'Water Flow', subcategory, detail
    elif 'water pump' in msg and 'pressure' in msg:
        subcategory = 'Pressure'
        if 'high' in msg or 'above' in msg:
            detail = 'High Pressure'
        elif 'low' in msg or 'below' in msg:
            detail = 'Low Pressure'
        else:
            detail = 'Pressure Issue'
        return 'Water Pressure', subcategory, detail
    elif 'membrane' in msg or 'filtrate' in msg:
        if 'scour' in msg:
            subcategory = 'Membrane Scour'
            detail = 'Scour Issue'
        else:
            subcategory = 'Filtration'
            detail = 'Filtration Issue'
        return 'Membrane & Filtration', subcategory, detail
    elif 'uv chamber' in msg or 'ultraviolet' in msg:
        subcategory = 'UV System'
        if 'dose' in msg:
            detail = 'Dosage Issue'
        else:
            detail = 'UV Issue'
        return 'UV Treatment', subcategory, detail
    return 'Other Systems', 'General', 'System Issue'

def load_and_process_data():
    """Load and process alarm data"""
    try:
        # Load data
        alarms_df = pd.read_csv('data/Alarms.csv')
        
        # Process timestamps
        alarms_df['timestamp'] = pd.to_datetime(alarms_df['timestamp'], dayfirst=True)
        
        # Add categorization
        categories = alarms_df['message'].apply(categorize_alarm)
        alarms_df['category'] = [x[0] for x in categories]
        alarms_df['subcategory'] = [x[1] for x in categories]
        alarms_df['detail'] = [x[2] for x in categories]
        
        # Add severity
        def determine_severity(msg):
            msg = msg.lower()
            if 'high' in msg or 'above' in msg:
                return 3  # High severity
            elif 'low' in msg or 'below' in msg:
                return 2  # Medium severity
            return 1  # Low severity
        
        alarms_df['severity_score'] = alarms_df['message'].apply(determine_severity)
        alarms_df['severity'] = alarms_df['severity_score'].map({
            3: 'High',
            2: 'Medium',
            1: 'Low'
        })
        
        return alarms_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        raise e

# Load data
df = load_and_process_data()

# Title and description
st.title('🚨 Enhanced Alarm Analysis Dashboard')
st.markdown("""
This enhanced dashboard provides both high-level insights and detailed analysis of system alarms.
Toggle between views to see different levels of detail.
""")

# Add engineer view toggle in sidebar
st.sidebar.title('Dashboard Settings')
view_mode = st.sidebar.toggle('Engineer View', value=False)

# Top metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Total Alarms", len(df))
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Unique Alarm Types", df['message'].nunique())
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Most Common Category", df['category'].mode()[0])
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    days_span = (df['timestamp'].max() - df['timestamp'].min()).days
    avg_alarms_per_day = len(df) / max(days_span, 1)
    st.metric("Avg Alarms/Day", f"{avg_alarms_per_day:.1f}")
    st.markdown('</div>', unsafe_allow_html=True)

if view_mode == True:
    st.markdown("---")
    st.subheader("Detailed Alarm Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Detailed sunburst chart
        alarm_hierarchy = df.groupby(
            ['category', 'subcategory', 'detail', 'severity']
        ).size().reset_index(name='count')
        
        fig_sunburst = px.sunburst(
            alarm_hierarchy,
            path=['category', 'subcategory', 'detail', 'severity'],
            values='count',
            title='Detailed Alarm Hierarchy',
            color='count',
            color_continuous_scale='Viridis'
        )
        fig_sunburst.update_layout(height=600)
        st.plotly_chart(fig_sunburst, use_container_width=True)
    
    with col2:
        # Category-specific metrics
        st.markdown("### Category Metrics")
        category_metrics = df.groupby('category').agg({
            'severity_score': ['mean', 'max'],
            'message': 'count'
        }).round(2)
        category_metrics.columns = ['Avg Severity', 'Max Severity', 'Count']
        st.dataframe(category_metrics)
        
        # Show severity distribution
        st.markdown("### Severity Distribution")
        severity_dist = df['severity'].value_counts()
        fig_severity = px.pie(
            values=severity_dist.values,
            names=severity_dist.index,
            title='Alarm Severity Distribution',
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig_severity, use_container_width=True)
    
    # Show filtered data table
    st.markdown("### Alarm Details")
    # Create a color-coded dataframe without using gradient
    styled_df = df[['timestamp', 'category', 'subcategory', 'detail', 'severity']].sort_values('timestamp', ascending=False)
    st.dataframe(styled_df, height=400)

    # Additional technical insights
    st.markdown("### Technical Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        # Show alarm patterns
        hourly_patterns = df.groupby([
            df['timestamp'].dt.hour,
            'category'
        ]).size().unstack(fill_value=0)
        
        fig_patterns = px.imshow(
            hourly_patterns,
            title='Alarm Patterns by Hour',
            labels=dict(x='Category', y='Hour of Day'),
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_patterns, use_container_width=True)
    
    with col2:
        # Show alarm correlations
        alarm_correlations = df.pivot_table(
            index='category',
            columns='severity',
            values='severity_score',
            aggfunc='count',
            fill_value=0
        )
        
        fig_correlations = px.imshow(
            alarm_correlations,
            title='Category vs Severity Correlation',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_correlations, use_container_width=True)

else:
    # Summary view
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Alarm Distribution Over Time")
        daily_alarms = df.groupby(['timestamp', 'category']).size().reset_index(name='count')
        fig_time = px.line(
            daily_alarms,
            x='timestamp',
            y='count',
            color='category',
            title='Alarms by Category Over Time'
        )
        fig_time.update_layout(height=400)
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        st.subheader("Category Distribution")
        basic_sunburst = df.groupby(['category', 'severity']).size().reset_index(name='count')
        fig_basic_sunburst = px.sunburst(
            basic_sunburst,
            path=['category', 'severity'],
            values='count',
            title='Alarm Categories and Severity'
        )
        fig_basic_sunburst.update_layout(height=400)
        st.plotly_chart(fig_basic_sunburst, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    💡 **Tips:**
    - In Technical view, click on categories in the sunburst chart to drill down
    - Hover over charts for detailed information
    - Use the severity distribution to identify critical areas
""")