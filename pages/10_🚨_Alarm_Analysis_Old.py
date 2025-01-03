import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
from pathlib import Path
import plotly.subplots as sp

# Add the root directory to Python path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

# Page config
st.set_page_config(page_title="Alarm & Warning Analysis", page_icon="🚨", layout="wide")

def load_alarm_data():
    """Load and process alarm data"""
    try:
        alarms_df = pd.read_csv('data/Alarms.csv')
        warnings_df = pd.read_csv('data/Warnings.csv')
        events_df = pd.read_csv('data/Events.csv')
        sequences_df = pd.read_csv('data/Sequences.csv')
        
        # Process timestamps
        for df in [alarms_df, warnings_df, events_df, sequences_df]:
            df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
            
        return alarms_df, warnings_df, events_df, sequences_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        raise e

# Load data
alarms_df, warnings_df, events_df, sequences_df = load_alarm_data()

# Title and description
st.title('🚨 Alarm & Warning Analysis')
st.markdown("""
This page provides a detailed analysis of system alarms and warnings during the Point Leo trial.
The analysis helps identify patterns, recurring issues, and potential areas for system optimization.
""")

# Create layout
col1, col2 = st.columns(2)

# Alarm Type Distribution
with col1:
    st.subheader('Alarm Type Distribution')
    
    # Count frequency of each alarm type
    alarm_counts = alarms_df['message'].value_counts()
    
    # Create pie chart
    fig_alarm_dist = go.Figure(data=[go.Pie(
        labels=alarm_counts.index,
        values=alarm_counts.values,
        hole=0.4,
        textinfo='percent+label'
    )])
    
    fig_alarm_dist.update_layout(
        height=400,
        title='Distribution of Alarm Types',
        showlegend=False
    )
    
    st.plotly_chart(fig_alarm_dist, use_container_width=True)

# Warning Type Distribution
with col2:
    st.subheader('Warning Type Distribution')
    
    # Count frequency of each warning type
    warning_counts = warnings_df['message'].value_counts()
    
    # Create pie chart
    fig_warning_dist = go.Figure(data=[go.Pie(
        labels=warning_counts.index,
        values=warning_counts.values,
        hole=0.4,
        textinfo='percent+label'
    )])
    
    fig_warning_dist.update_layout(
        height=400,
        title='Distribution of Warning Types',
        showlegend=False
    )
    
    st.plotly_chart(fig_warning_dist, use_container_width=True)

# Time Distribution Analysis
st.subheader('Time Distribution of Alarms and Warnings')

# Create figure with secondary y-axis
fig_time = sp.make_subplots(specs=[[{"secondary_y": True}]])

# Add traces for alarms and warnings
alarms_time = alarms_df.set_index('timestamp').resample('1H').size()
warnings_time = warnings_df.set_index('timestamp').resample('1H').size()

fig_time.add_trace(
    go.Scatter(x=alarms_time.index, y=alarms_time.values, name="Alarms",
               line=dict(color='red')),
    secondary_y=False,
)

fig_time.add_trace(
    go.Scatter(x=warnings_time.index, y=warnings_time.values, name="Warnings",
               line=dict(color='orange')),
    secondary_y=True,
)

# Update layout
fig_time.update_layout(
    title='Time Distribution of Alarms and Warnings',
    height=400,
    hovermode='x unified'
)

fig_time.update_xaxes(title_text="Time")
fig_time.update_yaxes(title_text="Number of Alarms", secondary_y=False)
fig_time.update_yaxes(title_text="Number of Warnings", secondary_y=True)

st.plotly_chart(fig_time, use_container_width=True)

# State Correlation Analysis
st.subheader('Correlation with System States')

# Merge alarms with sequences
merged_df = pd.merge_asof(
    alarms_df.sort_values('timestamp'),
    sequences_df[['timestamp', 'message']].sort_values('timestamp'),
    on='timestamp',
    direction='backward'
)

merged_df = merged_df.rename(columns={'message_x': 'alarm', 'message_y': 'state'})

# Create heatmap of alarms vs states
alarm_state_matrix = pd.crosstab(merged_df['alarm'], merged_df['state'])

fig_correlation = go.Figure(data=go.Heatmap(
    z=alarm_state_matrix.values,
    x=alarm_state_matrix.columns,
    y=alarm_state_matrix.index,
    colorscale='RdYlBu_r'
))

fig_correlation.update_layout(
    title='Correlation between Alarms and System States',
    height=600,
    xaxis_title='System State',
    yaxis_title='Alarm Type',
    xaxis={'tickangle': 45}
)

st.plotly_chart(fig_correlation, use_container_width=True)

# Top Recurring Issues
st.subheader('Top Recurring Issues')

# Combine alarms and warnings
all_issues = pd.concat([
    alarms_df[['message', 'timestamp']].assign(type='Alarm'),
    warnings_df[['message', 'timestamp']].assign(type='Warning')
])

# Calculate issue frequency
issue_counts = all_issues.groupby(['message', 'type']).size().reset_index(name='count')
issue_counts = issue_counts.sort_values('count', ascending=False)

# Create bar chart
fig_issues = go.Figure(data=[
    go.Bar(
        x=issue_counts['message'],
        y=issue_counts['count'],
        marker_color=issue_counts['type'].map({'Alarm': 'red', 'Warning': 'orange'})
    )
])

fig_issues.update_layout(
    title='Top Recurring Issues',
    height=500,
    xaxis_title='Issue Description',
    yaxis_title='Frequency',
    xaxis={'tickangle': 45}
)

st.plotly_chart(fig_issues, use_container_width=True)

# Key Insights
st.subheader('Key Insights')

# Calculate some statistics
total_alarms = len(alarms_df)
total_warnings = len(warnings_df)
most_common_alarm = alarm_counts.index[0]
most_common_warning = warning_counts.index[0]

# Create three columns for metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Alarms", total_alarms)
    st.markdown(f"**Most Common Alarm:**  \n{most_common_alarm}")

with col2:
    st.metric("Total Warnings", total_warnings)
    st.markdown(f"**Most Common Warning:**  \n{most_common_warning}")

with col3:
    alarm_per_day = total_alarms / len(alarms_df['timestamp'].dt.date.unique())
    st.metric("Average Alarms per Day", f"{alarm_per_day:.1f}")
    
# Add explanation
st.markdown("""
### Analysis Summary
- The system shows distinct patterns in alarm occurrences, with certain types being more frequent
- Water pump-related issues appear to be the most common type of alarm
- There is a correlation between certain system states and alarm frequencies
- The temporal distribution shows some clustering of alarms, suggesting potential systemic issues
during specific periods
""")