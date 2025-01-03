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
    page_title="Warning Analysis", 
    page_icon="⚠️", 
    layout="wide"
)

def categorize_warning(message):
    """Categorize warnings into main groups"""
    msg = message.lower()
    if 'water pump flow' in msg:
        subcategory = 'Flow Rate'
        if 'high' in msg:
            detail = 'High Flow'
        elif 'low' in msg:
            detail = 'Low Flow'
        else:
            detail = 'Flow Issue'
        return 'Water Flow', subcategory, detail
    elif 'water pump' in msg and 'pressure' in msg:
        subcategory = 'Pressure'
        if 'high' in msg:
            detail = 'High Pressure'
        elif 'low' in msg:
            detail = 'Low Pressure'
        else:
            detail = 'Pressure Issue'
        return 'Water Pressure', subcategory, detail
    elif 'membrane scour' in msg:
        subcategory = 'Membrane Scour'
        if 'high' in msg:
            detail = 'High Pressure'
        elif 'low' in msg:
            detail = 'Low Pressure'
        else:
            detail = 'Pressure Issue'
        return 'Membrane & Filtration', subcategory, detail
    elif 'clean-in-place' in msg:
        subcategory = 'CIP System'
        if 'high' in msg:
            detail = 'High Level'
        elif 'low' in msg:
            detail = 'Low Level'
        else:
            detail = 'Level Issue'
        return 'CIP System', subcategory, detail
    elif 'verify' in msg:
        subcategory = 'Verification'
        if 'uvm101' in msg:
            detail = 'UV System'
        elif 'vbl101' in msg:
            detail = 'Valve System'
        else:
            detail = 'General Verification'
        return 'System Verification', subcategory, detail
    return 'Other Systems', 'General', 'System Issue'

def load_and_process_data():
    """Load and process warning data"""
    try:
        # Load data
        warnings_df = pd.read_csv('data/Warnings.csv')
        
        # Process timestamps
        warnings_df['timestamp'] = pd.to_datetime(warnings_df['timestamp'], dayfirst=True)
        
        # Add categorization
        categories = warnings_df['message'].apply(categorize_warning)
        warnings_df['category'] = [x[0] for x in categories]
        warnings_df['subcategory'] = [x[1] for x in categories]
        warnings_df['detail'] = [x[2] for x in categories]
        
        # Add impact level
        def determine_impact(msg):
            msg = msg.lower()
            if 'high' in msg or 'failed' in msg:
                return 3  # High impact
            elif 'low' in msg:
                return 2  # Medium impact
            return 1  # Low impact
        
        warnings_df['impact_score'] = warnings_df['message'].apply(determine_impact)
        warnings_df['impact'] = warnings_df['impact_score'].map({
            3: 'High',
            2: 'Medium',
            1: 'Low'
        })
        
        return warnings_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        raise e

# Load data
df = load_and_process_data()

# Title and description
st.title('⚠️ Enhanced Warning Analysis Dashboard')
st.markdown("""
This enhanced dashboard provides both high-level insights and detailed analysis of system warnings.
Toggle between views to see different levels of detail.
""")

# Add engineer view toggle in sidebar
st.sidebar.title('Dashboard Settings')
view_mode = st.sidebar.toggle('Engineer View', value=False)

# Top metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Total Warnings", len(df))
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Unique Warning Types", df['message'].nunique())
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Most Common Category", df['category'].mode()[0])
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    days_span = (df['timestamp'].max() - df['timestamp'].min()).days
    avg_warnings_per_day = len(df) / max(days_span, 1)
    st.metric("Avg Warnings/Day", f"{avg_warnings_per_day:.1f}")
    st.markdown('</div>', unsafe_allow_html=True)

if view_mode == True:
    st.markdown("---")
    st.subheader("Detailed Warning Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create hierarchy with actual warning messages at the bottom level
        warning_hierarchy = df.groupby(
            ['category', 'subcategory', 'detail', 'message']
        ).size().reset_index(name='count')
        
        fig_sunburst = px.sunburst(
            warning_hierarchy,
            path=['category', 'subcategory', 'detail', 'message'],
            values='count',
            title='Detailed Warning Hierarchy',
            maxdepth=4  # Ensure all levels are shown
        )
        
        # Customize the layout for better readability
        fig_sunburst.update_layout(
            height=700,  # Increased height for better visibility
            # Adjust text size and wrapping
            uniformtext=dict(minsize=10, mode='hide'),
            # Ensure the chart uses full width
            width=None
        )
        
        # Update traces for better text wrapping
        fig_sunburst.update_traces(
            textinfo="label+value",
            insidetextorientation='radial'
        )
        
        st.plotly_chart(fig_sunburst, use_container_width=True)
    
    # Show filtered data table
    st.markdown("### Warning Details")
    styled_df = df[['timestamp', 'category', 'subcategory', 'detail', 'impact']].sort_values('timestamp', ascending=False)
    st.dataframe(styled_df, height=400)

    # Additional technical insights
    st.markdown("### Technical Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        # Show warning patterns
        hourly_patterns = df.groupby([
            df['timestamp'].dt.hour,
            'category'
        ]).size().unstack(fill_value=0)
        
        fig_patterns = px.imshow(
            hourly_patterns,
            title='Warning Patterns by Hour',
            labels=dict(x='Category', y='Hour of Day'),
            color_continuous_scale='YlOrRd'
        )
        st.plotly_chart(fig_patterns, use_container_width=True)
    
    with col2:
        # Show warning correlations
        warning_correlations = df.pivot_table(
            index='category',
            columns='impact',
            values='impact_score',
            aggfunc='count',
            fill_value=0
        )
        
        fig_correlations = px.imshow(
            warning_correlations,
            title='Category vs Impact Correlation',
            color_continuous_scale='YlOrRd'
        )
        st.plotly_chart(fig_correlations, use_container_width=True)

else:
    # Summary view
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Warning Distribution Over Time")
        daily_warnings = df.groupby(['timestamp', 'category']).size().reset_index(name='count')
        fig_time = px.line(
            daily_warnings,
            x='timestamp',
            y='count',
            color='category',
            title='Warnings by Category Over Time'
        )
        fig_time.update_layout(height=400)
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        st.subheader("Category Distribution")
        basic_sunburst = df.groupby(['category', 'impact']).size().reset_index(name='count')
        fig_basic_sunburst = px.sunburst(
            basic_sunburst,
            path=['category', 'impact'],
            values='count',
            title='Warning Categories and Impact',
            color_continuous_scale='YlOrRd'
        )
        fig_basic_sunburst.update_layout(height=400)
        st.plotly_chart(fig_basic_sunburst, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    💡 **Tips:**
    - In Technical view, click on categories in the sunburst chart to drill down
    - Hover over charts for detailed information
    - Use the impact distribution to identify areas requiring attention
""")