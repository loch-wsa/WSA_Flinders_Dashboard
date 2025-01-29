import streamlit as st
import sys
from pathlib import Path

# Add the root directory to Python path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from utils.data_loader import load_all_data, RELEVANT_PARAMS
from utils.charts import create_radar_chart

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

try:
    # Load data first to determine number of weeks
    influent_data, treated_data, influent_ranges, treated_ranges = get_water_data()
    
    # Determine available weeks from the data
    week_cols = [col for col in treated_data.columns if col.startswith('Week')]
    max_week = len(week_cols)

    # Sidebar controls
    st.sidebar.title('Control Panel')
    week_num = st.sidebar.slider('Select Week', 1, max_week, 1)
    show_all = st.sidebar.checkbox('Show All Parameters', value=False)

    # Get parameters based on selection
    # For comparison, we'll use the intersection of available parameters
    influent_params = influent_data['Influent Water'].tolist()
    treated_params = treated_data['Product Water'].tolist()
    available_params = list(set(influent_params) & set(treated_params))

    params = available_params if show_all else RELEVANT_PARAMS

    # Main content
    st.header('📊 Water Quality Comparison')
    st.markdown(f"""
    Week {week_num} comparison between influent and treated water.  
    The smaller radar plot area for treated water demonstrates the effectiveness of the Brolga treatment process.
    """)

    # Create and display comparison radar chart
    fig, warning = create_radar_chart(
        week_num, 
        params, 
        influent_data, 
        treated_data, 
        influent_ranges, 
        treated_ranges, 
        'influent',
        show_comparison=True
    )

    # Display warning if it exists
    if warning:
        st.warning(warning)

    st.plotly_chart(fig, use_container_width=True)

    # Add effectiveness metrics
    st.header('Treatment Effectiveness')
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Key Performance Indicators")
        
        # Calculate metrics for displayed parameters
        for param in params:
            try:
                influent_val = influent_data[influent_data['Influent Water'] == param][f'Week {week_num}'].values[0]
                treated_val = treated_data[treated_data['Product Water'] == param][f'Week {week_num}'].values[0]
                
                # Handle string values like '<0.001' or 'N/R'
                if isinstance(influent_val, str):
                    if influent_val.startswith('<'):
                        influent_val = float(influent_val.replace('<', ''))
                    elif 'LINT' in influent_val:
                        influent_val = float(influent_val.split()[0].replace('<', ''))
                    elif influent_val == 'N/R' or influent_val.strip() == '':
                        continue
                if isinstance(treated_val, str):
                    if treated_val.startswith('<'):
                        treated_val = float(treated_val.replace('<', ''))
                    elif 'LINT' in treated_val:
                        treated_val = float(treated_val.split()[0].replace('<', ''))
                    elif treated_val == 'N/R' or treated_val.strip() == '':
                        continue
                        
                influent_val = float(influent_val)
                treated_val = float(treated_val)
                
                # Special handling for pH
                if param.upper() == 'PH':
                    neutral_ph = 7.0
                    influent_diff = abs(influent_val - neutral_ph)
                    treated_diff = abs(treated_val - neutral_ph)
                    if influent_diff > 0:
                        improvement = ((influent_diff - treated_diff) / influent_diff) * 100
                        st.metric(
                            label=f"{param} (deviation from neutral)",
                            value=f"±{treated_diff:.2f}",
                            delta=f"{improvement:.1f}% improvement"
                        )
                # Standard reduction calculation for other parameters
                elif influent_val > 0:
                    reduction = ((influent_val - treated_val) / influent_val) * 100
                    st.metric(
                        label=param,
                        value=f"{treated_val:.2f}",
                        delta=f"{reduction:.1f}% reduction"
                    )
            except (ValueError, TypeError, IndexError) as e:
                st.warning(f"Unable to calculate metrics for {param}: Missing or invalid data")
                continue

    with col2:
        st.markdown("### Treatment Goals Achievement")
        
        try:
            # Get relevant parameter values for the current week
            ecoli_treated = treated_data[treated_data['Product Water'] == 'E COLI(C)'][f'Week {week_num}'].iloc[0]
            turbidity_treated = treated_data[treated_data['Product Water'] == 'TURBIDITY'][f'Week {week_num}'].iloc[0]
            ph_treated = treated_data[treated_data['Product Water'] == 'PH'][f'Week {week_num}'].iloc[0]
            toc_treated = treated_data[treated_data['Product Water'] == 'TOC'][f'Week {week_num}'].iloc[0]
            
            # Process values
            try:
                ecoli_ok = float(str(ecoli_treated).replace('<', '')) < 1
            except (ValueError, TypeError):
                ecoli_ok = False
                
            try:
                turbidity_ok = float(str(turbidity_treated).replace('<', '')) < 1
            except (ValueError, TypeError):
                turbidity_ok = False
                
            try:
                ph_ok = 6.5 <= float(ph_treated) <= 8.5
            except (ValueError, TypeError):
                ph_ok = False
                
            try:
                toc_ok = float(str(toc_treated).replace('<', '')) < 4
            except (ValueError, TypeError):
                toc_ok = False
            
            # Display achievement markers
            st.markdown("---")
            st.markdown(f"{'✓' if ecoli_ok else '⚠'} Pathogen removal targets met")
            st.markdown(f"{'✓' if turbidity_ok else '⚠'} Turbidity reduction achieved")
            st.markdown(f"{'✓' if ph_ok else '⚠'} pH within target range")
            st.markdown(f"{'✓' if toc_ok else '⚠'} Organic carbon reduction targets met")
            
        except Exception as e:
            st.warning("Unable to calculate treatment goals: Missing or invalid data")

    # Warning message
    st.sidebar.markdown('---')
    st.sidebar.warning('Note: Values below detection limits are shown as the detection limit value. Actual values may be lower.')

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.warning("Please check that all required data files are present in the data directory.")