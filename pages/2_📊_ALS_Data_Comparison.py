import streamlit as st
import sys
from pathlib import Path
import pandas as pd

# Add the root directory to Python path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from utils.data_loader import load_all_data
from utils.charts import create_radar_chart, create_parameter_table, map_parameters_to_ranges

# Page config
st.set_page_config(
    page_title="Water Quality Analysis Dashboard",
    page_icon="💧",
    layout="wide"
)

def determine_treated_column(data):
    """Determine which column name to use for treated water data"""
    possible_columns = ['Product Water', 'Treated Water', 'treated_water', 'product_water']
    available_columns = [col for col in possible_columns if col in data.columns]
    
    if available_columns:
        return available_columns[0]
    else:
        raise ValueError(f"No treated water column found. Available columns: {', '.join(data.columns)}")

def format_value(value):
    """Handle various string value formats and convert to float"""
    if isinstance(value, str):
        if value.startswith('<'):
            return float(value.replace('<', ''))
        elif 'LINT' in value:
            return float(value.split()[0].replace('<', ''))
        elif value == 'N/R' or value.strip() == '':
            return None
    return float(value)

def create_kpi_metrics(data_source, param_col, week_num, ranges_df, comparison_source=None):
    """Create KPI metric tiles with color-coded status indicators"""
    st.markdown("### Key Performance Indicators")
    
    # Filter out microbial parameters
    non_microbial_params = ranges_df[ranges_df['Category'] != 'Microbial'][param_col].unique()
    filtered_data = data_source[data_source[param_col].isin(non_microbial_params)]
    
    cols = st.columns(4)
    display_params = filtered_data[param_col].head(4)  # Show first 4 non-microbial parameters
    
    for idx, param in enumerate(display_params):
        try:
            # Get current value
            value = filtered_data[filtered_data[param_col] == param][f'Week {week_num}'].values[0]
            value = format_value(value)
                
            # Determine status and colors
            if comparison_source is not None:
                # Comparison mode
                influent_value = comparison_source[comparison_source['Influent Water'] == param][f'Week {week_num}'].values[0]
                influent_value = format_value(influent_value)
                
                reduction = ((influent_value - value) / influent_value * 100) if influent_value != 0 else 0
                is_improvement = reduction > 0
                color = "#22c55e" if is_improvement else "#ef4444"
                arrow = "↓" if is_improvement else "↑"
                status_text = f"{abs(reduction):.1f}% change"
            else:
                # Range check mode
                range_data = ranges_df[ranges_df[param_col] == param].iloc[0]
                min_val = format_value(range_data['Min'])
                max_val = format_value(range_data['Max'])
                
                is_in_range = min_val <= value <= max_val
                color = "#22c55e" if is_in_range else "#ef4444"
                arrow = "✓" if is_in_range else "⚠"
                status_text = "Within range" if is_in_range else "Out of range"
            
            # Add unit to display
            unit = ranges_df[ranges_df[param_col] == param]['Unit'].iloc[0]
            unit_text = f" {unit}" if pd.notna(unit) else ""
            
            # Create styled metric
            with cols[idx]:
                st.markdown(f"""
                <div style='padding: 1rem; border-radius: 0.5rem; text-align: center;'>
                    <h4 style='margin: 0; color: {color};'>{param}</h4>
                    <p style='font-size: 1.5rem; margin: 0.5rem 0; color: {color};'>{value:.2f}{unit_text}</p>
                    <p style='color: {color}; margin: 0;'>{arrow} {status_text}</p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.warning(f"Error creating KPI for {param}: {str(e)}")
            continue

def create_treatment_kpis(data_source, treated_column, week_num, treated_ranges):
    """Create KPI metrics for treatment standards"""
    st.markdown("### Treatment Standards")
    
    # Get microbial and key parameters
    critical_params = {
        'E COLI(C)': {'limit': 1.0, 'comparison': '<', 'category': 'Microbial'},
        'TURBIDITY': {'limit': 1.0, 'comparison': '<', 'category': 'Physical'},
        'PH': {'min': 6.5, 'max': 8.5, 'category': 'Physical'},
        'TOC': {'limit': 4.0, 'comparison': '<', 'category': 'Organic Compound'}
    }
    
    try:
        cols = st.columns(4)
        col_idx = 0
        
        for param, criteria in critical_params.items():
            param_data = data_source[data_source[treated_column] == param]
            if param_data.empty:
                continue
                
            try:
                value = param_data[f'Week {week_num}'].iloc[0]
                value = format_value(value)
                
                # Get unit from ranges
                unit = treated_ranges[treated_ranges[treated_column] == param]['Unit'].iloc[0]
                unit_text = f" {unit}" if pd.notna(unit) else ""
                
                # Determine if value meets criteria
                if 'limit' in criteria:
                    is_ok = value < criteria['limit'] if criteria['comparison'] == '<' else value > criteria['limit']
                    status_text = f"{'Below' if criteria['comparison'] == '<' else 'Above'} {criteria['limit']}"
                else:
                    is_ok = criteria['min'] <= value <= criteria['max']
                    status_text = f"Range: {criteria['min']} - {criteria['max']}"
                
                color = "#22c55e" if is_ok else "#ef4444"
                
                with cols[col_idx]:
                    st.markdown(f"""
                    <div style='padding: 1rem; border-radius: 0.5rem; text-align: center;'>
                        <h4 style='margin: 0; color: {color};'>{param}</h4>
                        <p style='font-size: 1.5rem; margin: 0.5rem 0; color: {color};'>{value:.2f}{unit_text}</p>
                        <p style='color: {color}; margin: 0;'>{'✓' if is_ok else '⚠'} {status_text}</p>
                    </div>
                    """, unsafe_allow_html=True)
                col_idx += 1
                    
            except Exception as e:
                st.warning(f"Error processing {param}: {str(e)}")
                continue
            
    except Exception as e:
        st.warning("Unable to calculate treatment standards: Missing or invalid data")

# Cache data loading at the application level
@st.cache_data(ttl=3600)
def get_water_data():
    """Load and prepare water quality data"""
    try:
        all_data = load_all_data()
        
        # Map parameters to ranges
        all_data['influent_ranges'] = map_parameters_to_ranges(all_data['influent_data'], all_data['influent_params'], 'Influent')
        all_data['treated_ranges'] = map_parameters_to_ranges(all_data['treated_data'], all_data['treated_params'], 'Treated')
        
        return all_data
    except Exception as e:
        st.error(f"Error in get_water_data: {str(e)}")
        raise
        
def main():
    try:
        # Load all data
        data = get_water_data()
        
        # Map parameters to ranges using new structure
        influent_ranges = map_parameters_to_ranges(data['influent_data'], data['influent_params'], 'Influent')
        treated_ranges = map_parameters_to_ranges(data['treated_data'], data['treated_params'], 'Treated')
        
        # Determine treated water column name
        try:
            treated_column = determine_treated_column(data['treated_data'])
        except ValueError as e:
            st.error(f"Error determining treated water column: {str(e)}")
            st.stop()

        # Determine available weeks from the data
        week_cols = [col for col in data['treated_data'].columns if col.startswith('Week')]
        max_week = len(week_cols)

        # Sidebar controls
        st.sidebar.title('Control Panel')
        week_num = st.sidebar.slider('Select Week', 1, max_week, 1)
        show_raw_params = st.sidebar.checkbox('Show Raw Parameters Table', value=False)
        use_brolga_limits = st.sidebar.checkbox('Use Brolga Limits For Range', value=True)

        # Create tabs
        tab1, tab2, tab3 = st.tabs(["🚱 Influent Water", "🚰 Treated Water", "📊 Comparison"])

        # Tab 1: Influent Water Analysis
        with tab1:
            st.header('🚱 Influent Water Analysis')
            st.markdown(f"""
            Analysing raw pond water characteristics for Week {week_num}.  
            The data represents untreated water entering the Brolga system.
            """)

            try:
                fig, warning = create_radar_chart(
                    week_num,
                    data['influent_data'],
                    data['treated_data'],
                    data['influent_params'],
                    data['treated_params'],
                    'influent',  # or 'treated' for treated water
                    show_comparison=False,
                    use_brolga_limits=True
                )

                if warning:
                    st.warning(warning)

                st.plotly_chart(fig, use_container_width=True)
                create_kpi_metrics(data['influent_data'], 'Influent Water', week_num, influent_ranges)

                if show_raw_params:
                    st.markdown("### Raw Water Parameters")
                    non_microbial_params = influent_ranges[influent_ranges['Category'] != 'Microbial']['Influent Water'].tolist()
                    df_display = create_parameter_table(
                        week_num,
                        non_microbial_params,
                        data['influent_data'],  # or data['treated_data']
                        data['influent_params']  # or data['treated_params']
                    )
                    st.dataframe(df_display)
            except Exception as e:
                st.error(f"Error displaying influent water analysis: {str(e)}")

        # Tab 2: Treated Water Analysis
        with tab2:
            st.header('🚰 Potable Water Analysis')
            st.markdown(f"""
            Showing treated water quality parameters for Week {week_num}.  
            This represents the Brolga system's output water quality after full treatment.
            """)

            try:
                fig, warning = create_radar_chart(
                    week_num,
                    data['influent_data'],
                    data['treated_data'],
                    influent_ranges,
                    treated_ranges,
                    'treated',
                    use_brolga_limits=use_brolga_limits
                )

                if warning:
                    st.warning(warning)

                st.plotly_chart(fig, use_container_width=True)
                
                create_kpi_metrics(data['treated_data'], treated_column, week_num, treated_ranges)
                create_treatment_kpis(data['treated_data'], treated_column, week_num, treated_ranges)

                if show_raw_params:
                    st.markdown("### Treated Water Parameters")
                    non_microbial_params = treated_ranges[treated_ranges['Category'] != 'Microbial'][treated_column].tolist()
                    df_display = create_parameter_table(week_num, non_microbial_params, data['treated_data'], treated_ranges)
                    st.dataframe(df_display)
            except Exception as e:
                st.error(f"Error displaying treated water analysis: {str(e)}")

        # Tab 3: Comparison
        with tab3:
            st.header('📊 Water Quality Comparison')
            st.markdown(f"""
            Week {week_num} comparison between influent and treated water.  
            The smaller radar plot area for treated water demonstrates the effectiveness of the Brolga treatment process.
            """)

            try:
                fig, warning = create_radar_chart(
                    week_num,
                    data['influent_data'],
                    data['treated_data'],
                    influent_ranges,
                    treated_ranges,
                    'influent',
                    show_comparison=True,
                    use_brolga_limits=use_brolga_limits
                )

                if warning:
                    st.warning(warning)

                st.plotly_chart(fig, use_container_width=True)
                
                create_kpi_metrics(
                    data['treated_data'],
                    treated_column,
                    week_num,
                    treated_ranges,
                    comparison_source=data['influent_data']
                )
                create_treatment_kpis(data['treated_data'], treated_column, week_num, treated_ranges)

                if show_raw_params:
                    st.markdown("### Detailed Parameters")
                    non_microbial_params = treated_ranges[treated_ranges['Category'] != 'Microbial'][treated_column].tolist()
                    df_display = create_parameter_table(week_num, non_microbial_params, data['treated_data'], treated_ranges)
                    st.dataframe(df_display)
            except Exception as e:
                st.error(f"Error displaying comparison analysis: {str(e)}")

        # Warning message in sidebar
        st.sidebar.markdown('---')
        st.sidebar.warning('Note: Values below detection limits are shown as the detection limit value. Actual values may be lower.')

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.warning("Please check that all required data files are present in the data directory and have the correct format.")
        st.stop()

if __name__ == "__main__":
    main()