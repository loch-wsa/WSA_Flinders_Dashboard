import streamlit as st
import pandas as pd
from datetime import timedelta
import plotly.graph_objects as go
import os
import sys
from functools import lru_cache

# Add the utils directory to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.append(utils_path)

# Import the function from ranged_charts.py
from ranged_charts import plot_sensors
from init import initialize_date_range

# Cache the data loading function
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_and_process_telemetry():
    """Load and process telemetry data with caching"""
    df = pd.read_csv('data/Telemetry.csv', parse_dates=['TIMESTAMP'])
    return df

# Cache the thresholds loading
@st.cache_data(ttl=3600)
def load_thresholds():
    """Load threshold data with caching"""
    return pd.read_csv('data/Thresholds.csv')

# Cache component processing
@st.cache_data(ttl=3600)
def process_available_components(telemetry_df, thresholds_df):
    """Process and return available components with caching"""
    # Get telemetry columns excluding 'TIMESTAMP'
    telemetry_columns = set(col for col in telemetry_df.columns if col != 'TIMESTAMP')
    
    # Get unique components from thresholds dataframe
    threshold_components = set(thresholds_df['Component'].unique())
    
    # Find intersection while preserving case
    available_components = telemetry_columns.intersection(threshold_components)
    
    # Return sorted list of components
    return sorted(available_components)

# Cache threshold processing for components
@st.cache_data(ttl=3600)
def get_component_thresholds(thresholds_df, component):
    """Get thresholds for a specific component with caching"""
    threshold = thresholds_df[thresholds_df['Component'] == component]
    if not threshold.empty:
        return {
            'high_high': threshold['HighHigh'].values[0],
            'high': threshold['High'].values[0],
            'low': threshold['Low'].values[0],
            'low_low': threshold['LowLow'].values[0]
        }
    return {
        'high_high': None,
        'high': None,
        'low': None, 
        'low_low': None
    }

@st.cache_data(ttl=3600)
def remove_outliers(data, column):
    """Remove outliers using the IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
def main():
    # Title and description
    st.title("📊 Sensor Data Overview")
    st.markdown("""
        This page provides line charts for various sensor measurements over time. Each chart represents data from a specific sensor.
        Use the controls below to adjust the time range and view specific periods of interest.
    """)

    with st.sidebar:
        st.header("Configuration")
        remove_outliers_enabled = st.toggle("Remove Outliers", value=False)
        
        if remove_outliers_enabled:
            st.info("""
                **Outlier Removal Formula:**
                - Calculate Q1 (25th percentile) and Q3 (75th percentile)
                - Calculate IQR = Q3 - Q1
                - Remove data points outside range:
                  - Lower bound = Q1 - 1.5 × IQR
                  - Upper bound = Q3 + 1.5 × IQR
                
                This is known as the Interquartile Range (IQR) method.
            """)

    try:
        # Initialize session state for configurations
        if 'init' not in st.session_state:
            st.session_state.init = True
            st.session_state.show_raw_data = False
        
        # Load data using cached functions
        with st.spinner('Loading telemetry data...'):
            telemetry_df = load_and_process_telemetry()
            thresholds_df = load_thresholds()
        
        # Initialize date range controls and get filtered data
        filtered_telemetry_df, display_format = initialize_date_range(telemetry_df)
        
        # Get available components using cached function
        available_components = process_available_components(filtered_telemetry_df, thresholds_df)
        
        # Component selector with multiselect checkboxes
        selected_components = st.multiselect(
            'Select Components to Display',
            options=available_components,
            default=['PTC109_PRESSURE'] if 'PTC109_PRESSURE' in available_components else []
        )
        
        # Process selected components efficiently
        for component in selected_components:
            st.subheader(f"{component} Sensor Readings")
            
            # Get component data
            component_data = filtered_telemetry_df[['TIMESTAMP', component]].dropna()
            
            if remove_outliers_enabled:
                original_count = len(component_data)
                component_data = remove_outliers(component_data, component)
                removed_count = original_count - len(component_data)
                if removed_count > 0:
                    st.caption(f"Removed {removed_count} outliers from {component}")
            
            if not component_data.empty:
                df_dict = {
                    component: component_data.rename(
                        columns={'timestamp': 'Timestamp', component: 'Value'}
                    )
                }
                
                # Get thresholds using cached function
                thresholds = get_component_thresholds(thresholds_df, component)
                
                # Plot the component data
                plot_sensors(
                    df_dict,
                    high_high_threshold=thresholds['high_high'],
                    high_threshold=thresholds['high'],
                    low_threshold=thresholds['low'],
                    low_low_threshold=thresholds['low_low']
                )
        
        # Toggle for raw data visibility
        st.session_state.show_raw_data = st.toggle('Show Raw Data', st.session_state.show_raw_data)
        
        # Display raw data in an expandable section if toggled
        if st.session_state.show_raw_data:
            st.write("Telemetry Data Sample (last 5 records):")
            st.dataframe(
                filtered_telemetry_df[['TIMESTAMP'] + selected_components].tail(),
                use_container_width=True
            )
            
            st.write("\nThreshold Settings for Selected Components:")
            st.dataframe(
                thresholds_df[thresholds_df['Component'].isin(selected_components)],
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Error loading or processing data: {str(e)}")
        st.write("Please check that both 'Telemetry.csv' and 'Thresholds.csv' files are in the correct location")
        st.write("Error details:", str(e))

if __name__ == "__main__":
    st.set_page_config(
        page_title="Telemetry Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    main()