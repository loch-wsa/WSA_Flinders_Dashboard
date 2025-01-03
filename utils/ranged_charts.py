import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st


def load_asset_info():
    """
    Loads the asset information from Assets.csv
    Returns a dictionary mapping sensor names to their display names and units
    """
    try:
        assets_df = pd.read_csv('data/Assets.csv')
        # Create a dictionary mapping sensor names to (display name, unit) tuples
        asset_info = {
            row['Sensor']: (row['Name'], row['Unit']) 
            for _, row in assets_df.iterrows()
        }
        return asset_info
    except Exception as e:
        st.error(f"Error loading Assets.csv: {str(e)}")
        return {}

def get_chart_labels(sensor_name, asset_info):
    """
    Gets the appropriate chart labels for a given sensor
    
    Parameters:
    - sensor_name (str): The sensor identifier
    - asset_info (dict): Dictionary mapping sensor names to (display name, unit) tuples
    
    Returns:
    - tuple: (title, y_axis_label)
    """
    if sensor_name in asset_info:
        display_name, unit = asset_info[sensor_name]
        title = f'{display_name} Readings with Thresholds'
        y_axis_label = f'{display_name} ({unit})'
    else:
        # Fallback to generic labels if sensor not found
        title = 'Sensor Readings with Thresholds'
        y_axis_label = 'Value Level'
    
    return title, y_axis_label

def plot_sensors(dataframes, high_high_threshold=None, high_threshold=None, 
                low_threshold=None, low_low_threshold=None):
    """
    Plots readings from multiple sensors with optional thresholds. Each threshold line and its
    associated shading will only appear if that specific threshold value is provided.
    
    Parameters:
    - dataframes (dict): Dictionary where keys are sensor names and values are DataFrames with 'TIMESTAMP' and 'Value' columns.
    - high_high_threshold (float, optional): Highest threshold for unsafe values
    - high_threshold (float, optional): Upper safe value limit
    - low_threshold (float, optional): Lower safe value limit
    - low_low_threshold (float, optional): Lowest threshold for unsafe values
    """
    
    # Load asset information
    asset_info = load_asset_info()
 
    fig = go.Figure()
    # Determine the full time range and y-axis range across all sensors
    all_timestamps = pd.concat([df['TIMESTAMP'] for df in dataframes.values()])
    x_min, x_max = all_timestamps.min(), all_timestamps.max()
    all_values = pd.concat([df['Value'] for df in dataframes.values()])
    y_min, y_max = all_values.min(), all_values.max()
    
    # Plot each sensor's data
    for sensor_name, df in dataframes.items():
        # Get display name from asset info if available
        display_name = asset_info[sensor_name][0] if sensor_name in asset_info else sensor_name
        
        fig.add_trace(go.Scatter(
            x=df['TIMESTAMP'],
            y=df['Value'],
            mode='lines',
            name=display_name,  # Use display name in legend
            line=dict(width=2)
        ))

    # Helper function to safely check if a threshold is valid
    def is_valid_threshold(threshold):
        if threshold is None:
            return False
        try:
            return not pd.isna(threshold) and isinstance(float(threshold), float)
        except (ValueError, TypeError):
            return False

    # Add shaded regions and lines for each provided threshold
    # HIGH zone (between high and high-high)
    if is_valid_threshold(high_threshold) and is_valid_threshold(high_high_threshold):
        fig.add_shape(type="rect",
                    xref="x", yref="y",
                    x0=x_min, y0=float(high_threshold),
                    x1=x_max, y1=float(high_high_threshold),
                    fillcolor="yellow", opacity=0.3, line_width=0, layer="below")

    # LOW zone (between low and low-low)
    if is_valid_threshold(low_threshold) and is_valid_threshold(low_low_threshold):
        fig.add_shape(type="rect",
                    xref="x", yref="y",
                    x0=x_min, y0=float(low_threshold),
                    x1=x_max, y1=float(low_low_threshold),
                    fillcolor="yellow", opacity=0.3, line_width=0, layer="below")
    
    # Add threshold lines independently
    if is_valid_threshold(high_high_threshold):
        fig.add_hline(y=float(high_high_threshold), line=dict(color="red", dash="dash"),
                    annotation_text="High High Threshold", annotation_position="top right")
    
    if is_valid_threshold(high_threshold):
        fig.add_hline(y=float(high_threshold), line=dict(color="yellow", dash="dash"),
                    annotation_text="High Threshold", annotation_position="top right")
    
    if is_valid_threshold(low_threshold):
        fig.add_hline(y=float(low_threshold), line=dict(color="yellow", dash="dash"),
                    annotation_text="Low Threshold", annotation_position="top right")
    
    if is_valid_threshold(low_low_threshold):
        fig.add_hline(y=float(low_low_threshold), line=dict(color="red", dash="dash"),
                    annotation_text="Low Low Threshold", annotation_position="top right")

    # Calculate y-axis range based on all valid threshold values
    valid_thresholds = [float(t) for t in [high_high_threshold, high_threshold, 
                                         low_threshold, low_low_threshold] 
                       if is_valid_threshold(t)]

    if valid_thresholds:
        y_range = [
            min(y_min, min(valid_thresholds)),
            max(y_max, max(valid_thresholds))
        ]
    else:
        y_range = [y_min, y_max]

    # Get appropriate chart labels based on the first sensor
    # (assuming all sensors in the chart are of the same type)
    first_sensor = next(iter(dataframes.keys()))
    title, y_axis_label = get_chart_labels(first_sensor, asset_info)
    
    # Update layout for clear display with dynamic labels
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title=y_axis_label,
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=y_range),
        showlegend=True,
        template="plotly_white"
    )
    
    # Display in Streamlit
    st.plotly_chart(fig)