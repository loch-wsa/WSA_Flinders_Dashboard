import streamlit as st
import sys
from pathlib import Path
from typing import Dict, List

# Add parameter groupings
PARAMETER_GROUPS = {
    "Physical Parameters": [
        "TURBIDITY",
        "TRUE COLOUR",
        "SUS SOLIDS",
        "TDS_180",
        "UV TRANS",
        "EC"
    ],
    "Basic Chemical Parameters": [
        "PH",
        "ALKALINITY AS CACO3",
        "CO3 AS CACO3",
        "HCO3 AS CACO3",
        "OH AS CACO3",
        "W-HARDNESS"
    ],
    "Organic Parameters": [
        "TOC",
        "DOC"
    ],
    "Inorganic Parameters": [
        "W-CHLORIDE(DA)",
        "W-SO4-DA",
        "W-NO3-N",
        "SILICA"
    ],
    "Metals": [
        "FE",
        "MN",
        "AS",
        "HG",
        "PB",
        "NA"
    ],
    "Microbiological Parameters": [
        "COLIFORM (C)",
        "E COLI(C)",
        "PLATE COUNT 36 C"
    ]
}


from utils.data_loader import load_data, RELEVANT_PARAMS
from utils.charts import create_radar_chart, create_parameter_table

# Page config
st.set_page_config(page_title="Influent Water Analysis", page_icon="🚱", layout="wide")

# Sidebar controls
st.sidebar.title('Control Panel')
week_num = st.sidebar.slider('Select Week', 1, 7, 1)

# Add group selection to sidebar
selected_groups = st.sidebar.multiselect(
    'Select Parameter Groups',
    options=list(PARAMETER_GROUPS.keys()),
    default=["Physical Parameters"]
)

# Get parameters based on selected groups
selected_params = []
for group in selected_groups:
    selected_params.extend(PARAMETER_GROUPS[group])

use_brolga_limits = st.sidebar.checkbox('Use Brolga Limits For Range', value=True)

# Load data with appropriate range settings
influent_data, treated_data, influent_ranges, treated_ranges = load_data(use_brolga_limits)

# Main content
st.header('🚱 Influent Water Analysis')
st.markdown(f"""
Analysing raw pond water characteristics for Week {week_num}.  
The data represents untreated water entering the Brolga system.
""")

# Create and display radar chart for selected parameters
fig, warning = create_radar_chart(
    week_num, 
    selected_params,
    influent_data, 
    treated_data, 
    influent_ranges, 
    treated_ranges, 
    'influent'
)

# Display warning if it exists
if warning:
    st.warning(warning)

# Display the chart
st.plotly_chart(fig, use_container_width=True)

# Display parameter tables by group
for group in selected_groups:
    st.markdown(f"### {group}")
    group_params = PARAMETER_GROUPS[group]
    df_display = create_parameter_table(week_num, group_params, influent_data, influent_ranges)
    st.dataframe(df_display)

# Warning message
st.sidebar.markdown('---')
st.sidebar.warning('Note: Values below detection limits are shown as the detection limit value. Actual values may be lower.')