import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Optional
import numpy as np

def create_threshold_chart(df: pd.DataFrame, 
                         component: str,
                         y_axis_label: Optional[str] = None,
                         height: int = 600) -> go.Figure:
    """
    Create a threshold visualization chart for a specific component.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the threshold data with columns:
        Component, Sequence, LowLow, Low, OpLow, OpHigh, High, HighHigh
    component : str
        The component to visualize
    y_axis_label : str, optional
        Custom label for y-axis
    height : int, optional
        Height of the chart in pixels
        
    Returns:
    --------
    go.Figure
        Plotly figure object with the threshold visualization
    """
    
    # Filter data for the specific component
    component_data = df[df['Component'] == component].copy()
    
    if len(component_data) == 0:
        raise ValueError(f"No data found for component: {component}")
    
    # Create figure
    fig = go.Figure()
    
    # Calculate y-axis range
    y_values = []
    for col in ['LowLow', 'Low', 'OpLow', 'OpHigh', 'High', 'HighHigh']:
        y_values.extend(component_data[col].dropna().tolist())
    
    y_min = min(y_values) if y_values else 0
    y_max = max(y_values) if y_values else 100
    
    # Add some padding to y-axis range
    y_padding = (y_max - y_min) * 0.1
    y_min -= y_padding
    y_max += y_padding
    
    # For each sequence in the data
    for idx, row in component_data.iterrows():
        x_pos = row['Sequence']
        
        # Add red region for LowLow
        if pd.notna(row['LowLow']) and pd.notna(row['Low']):
            fig.add_trace(go.Scatter(
                x=[x_pos, x_pos],
                y=[y_min, row['Low']],
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(width=0),
                showlegend=False,
                name='LowLow Region'
            ))
        
        # Add orange region for Low
        if pd.notna(row['Low']) and pd.notna(row['OpLow'] if pd.notna(row['OpLow']) else row['High']):
            fig.add_trace(go.Scatter(
                x=[x_pos, x_pos],
                y=[row['Low'], row['OpLow'] if pd.notna(row['OpLow']) else row['High']],
                fill='tonexty',
                fillcolor='rgba(255,165,0,0.2)',
                line=dict(width=0),
                showlegend=False,
                name='Low Region'
            ))
        
        # Add orange region for High
        if pd.notna(row['High']) and pd.notna(row['HighHigh']):
            fig.add_trace(go.Scatter(
                x=[x_pos, x_pos],
                y=[row['High'], row['HighHigh']],
                fill='tonexty',
                fillcolor='rgba(255,165,0,0.2)',
                line=dict(width=0),
                showlegend=False,
                name='High Region'
            ))
        
        # Add red region for HighHigh
        if pd.notna(row['HighHigh']):
            fig.add_trace(go.Scatter(
                x=[x_pos, x_pos],
                y=[row['HighHigh'], y_max],
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(width=0),
                showlegend=False,
                name='HighHigh Region'
            ))
        
        # Add OpLow line
        if pd.notna(row['OpLow']):
            fig.add_trace(go.Scatter(
                x=[x_pos],
                y=[row['OpLow']],
                mode='markers',
                marker=dict(
                    symbol='line-ns',
                    size=20,
                    color='red',
                    line=dict(width=2)
                ),
                name='OpLow',
                showlegend=False
            ))
        
        # Add OpHigh line
        if pd.notna(row['OpHigh']):
            fig.add_trace(go.Scatter(
                x=[x_pos],
                y=[row['OpHigh']],
                mode='markers',
                marker=dict(
                    symbol='line-ns',
                    size=20,
                    color='red',
                    line=dict(width=2)
                ),
                name='OpHigh',
                showlegend=False
            ))
    
    # Update layout
    fig.update_layout(
        title=f'Threshold Visualization for {component}',
        xaxis_title='Sequence',
        yaxis_title=y_axis_label or component,
        height=height,
        showlegend=False,
        yaxis=dict(range=[y_min, y_max]),
        xaxis=dict(
            type='category',
            categoryorder='array',
            categoryarray=component_data['Sequence'].tolist()
        )
    )
    
    return fig

# Example usage in Streamlit app
def main():
    st.title('Component Threshold Visualization')
    
    # Load your data
    df = pd.read_csv('thresholds.csv')
    
    # Get unique components
    components = sorted(df['Component'].unique())
    
    # Component selector
    selected_component = st.selectbox(
        'Select Component',
        components
    )
    
    # Create and display chart
    try:
        fig = create_threshold_chart(df, selected_component)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")

if __name__ == "__main__":
    main()