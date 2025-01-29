import streamlit as st
import sys
from pathlib import Path
import pandas as pd

# Add the root directory to Python path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from utils.data_loader import load_all_data
from utils.charts import create_radar_chart
from utils.tiles import create_parameter_tiles_grid, create_collapsible_section

# Page config
st.set_page_config(page_title="Water Quality Comparison", page_icon="📊", layout="wide")

# Cache data loading at the page level
@st.cache_data(ttl=3600)
def get_water_data():
    """Load and prepare water quality data"""
    all_data = load_all_data()
    return (
        all_data['influent_data'],
        all_data['treated_data'],
        all_data['influent_ranges'],
        all_data['treated_ranges']
    )

def display_microbial_section(data_df, ranges_df, week_num):
    """Display microbial parameters in tiles"""
    # Get all microbial parameters
    microbial_params = ranges_df[ranges_df['Category'] == 'Microbial']
    
    if not microbial_params.empty:
        params = []
        values = []
        statuses = []
        
        week_col = f'Week {week_num}'
        
        for _, row in microbial_params.iterrows():
            params.append(row['Parameter'])
            
            # Check if parameter has ALS Lookup and data
            if pd.notna(row['ALS Lookup']) and row['ALS Lookup'] != '':
                param_data = data_df[data_df['ALS Lookup'] == row['ALS Lookup']]
                
                if not param_data.empty:
                    value = param_data[week_col].iloc[0]
                    if pd.isna(value) or value == 'N/R':
                        values.append("Not Tested")
                        statuses.append('untested')
                    else:
                        try:
                            float_value = float(str(value).replace('<', ''))
                            values.append(f"{float_value:.1f} {row['Unit']}")
                            statuses.append('neutral')  # Use neutral for actual values
                        except ValueError:
                            values.append(str(value))
                            statuses.append('neutral')
                else:
                    values.append("Not Tested")
                    statuses.append('untested')
            else:
                values.append("Not Tested")
                statuses.append('untested')
        
        create_parameter_tiles_grid(params, values, statuses)

def display_category_section(category, data_df, treated_data, ranges_df, treated_ranges, week_num, influent_data=None, influent_ranges=None):
    """Display a category section with radar chart and parameter tiles"""
    # Handle different spellings of Disinfection By-Products
    if category == 'Disinfection By-Products':
        category_params = ranges_df[ranges_df['Category'].isin(['Disinfection By-Products', 'Dysenfection Bi-Products'])]
    else:
        category_params = ranges_df[ranges_df['Category'] == category]
    
    if not category_params.empty:
        # Split parameters into those with and without ALS Lookup
        params_with_lookup = category_params[pd.notna(category_params['ALS Lookup']) & 
                                          (category_params['ALS Lookup'] != '')]
        
        # Display radar chart for parameters with ALS Lookup if there are multiple parameters
        # or if it's the Organic Compound category
        if not params_with_lookup.empty and (len(params_with_lookup) > 1 or category == 'Organic Compound'):
            als_lookups = params_with_lookup['ALS Lookup'].tolist()
            
            # Use the provided influent data if available, otherwise use data_df
            source_data = influent_data if influent_data is not None else data_df
            source_ranges = influent_ranges if influent_ranges is not None else ranges_df
            
            fig, _ = create_radar_chart(
                week_num,
                als_lookups,
                source_data,
                treated_data,
                source_ranges,
                treated_ranges,
                'comparison',
                category
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Create tiles for all parameters in a collapsible section
        with st.expander(f"View {category} Parameters", expanded=False):
            week_col = f'Week {week_num}'
            params = []
            values = []
            statuses = []
            ranges_min = []
            ranges_max = []
            units = []
            
            # Process all parameters
            for _, row in category_params.iterrows():
                params.append(row['Parameter'])
                units.append(row['Unit'] if pd.notna(row['Unit']) else '')
                ranges_min.append(row['Min'] if pd.notna(row['Min']) else None)
                ranges_max.append(row['Max'] if pd.notna(row['Max']) else None)
                
                if pd.notna(row['ALS Lookup']) and row['ALS Lookup'] != '':
                    # Parameter has ALS Lookup
                    param_data = data_df[data_df['ALS Lookup'] == row['ALS Lookup']]
                    
                    if not param_data.empty:
                        value = param_data[week_col].iloc[0]
                        if pd.isna(value) or value == 'N/R':
                            values.append("Not Tested")
                            statuses.append('untested')
                        else:
                            try:
                                float_value = float(str(value).replace('<', ''))
                                values.append(float_value)
                                
                                # Check if value is within range
                                min_val = float(row['Min']) if pd.notna(row['Min']) else None
                                max_val = float(row['Max']) if pd.notna(row['Max']) else None
                                
                                if (min_val is not None and float_value < min_val) or \
                                   (max_val is not None and float_value > max_val):
                                    statuses.append('negative')
                                else:
                                    statuses.append('positive')
                            except ValueError:
                                values.append(str(value))
                                statuses.append('neutral')
                    else:
                        values.append("Not Tested")
                        statuses.append('untested')
                else:
                    # Parameter has no ALS Lookup
                    values.append("Not Tested")
                    statuses.append('untested')
            
            create_parameter_tiles_grid(
                parameters=params, 
                values=values, 
                statuses=statuses,
                ranges_min=ranges_min,
                ranges_max=ranges_max,
                units=units
            )

def app():
    try:
        # Load data first to determine number of weeks
        influent_data, treated_data, influent_ranges, treated_ranges = get_water_data()
        
        # Determine available weeks from the data
        week_cols = [col for col in treated_data.columns if col.startswith('Week')]
        max_week = len(week_cols)

        # Sidebar controls
        st.sidebar.title('Control Panel')
        week_num = st.sidebar.slider('Select Week', 1, max_week, 1)

        # Main content
        st.header('📊 Water Quality Comparison')
        st.markdown(f"""
        Week {week_num} comparison between influent and treated water.  
        The smaller radar plot area for treated water demonstrates the effectiveness of the Brolga treatment process.
        """)

        # Main categories section
        st.subheader('Main Parameters')
        main_categories = ['Inorganic Compound', 'Metal', 'Organic Compound', 'Physical']
        
        cols = st.columns(2)
        for i, category in enumerate(main_categories):
            with cols[i % 2]:
                st.markdown(f"#### {category}")
                display_category_section(
                    category,
                    treated_data,
                    treated_data,
                    treated_ranges,
                    treated_ranges,
                    week_num,
                    influent_data,  # Add influent_data as a new parameter
                    influent_ranges  # Add influent_ranges as a new parameter
                )

        # Microbial section
        st.subheader('Microbial Parameters')
        display_microbial_section(treated_data, treated_ranges, week_num)

        # Additional categories
        for category in ['Radiological', 'Disinfection By-Products', 'Algae Toxins']:
            st.subheader(f'{category} Parameters')
            display_category_section(
                category,
                treated_data,
                treated_data,
                treated_ranges,
                treated_ranges,
                week_num,
                influent_data,  # Add influent_data as a new parameter
                influent_ranges  # Add influent_ranges as a new parameter
            )

        # Warning message
        st.sidebar.markdown('---')
        st.sidebar.warning('Note: Values below detection limits are shown as the detection limit value. Actual values may be lower.')

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.warning("Please check that all required data files are present in the data directory.")

# Run the app
if __name__ == "__main__":
    app()