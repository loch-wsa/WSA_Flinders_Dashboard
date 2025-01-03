import streamlit as st
import sys
from pathlib import Path

# Add the root directory to Python path
root_dir = Path(__file__).parent
sys.path.append(str(root_dir))

from utils.data_loader import load_data, RELEVANT_PARAMS

# Page configuration
st.set_page_config(
    layout="wide",
    page_title="Brolga Water Treatment Trial - Flinders",
    page_icon="💧"
)

# Load data
influent_data, treated_data, influent_ranges, treated_ranges = load_data()

# Project Overview Section
st.title('Brolga Water Treatment System - Flinders Trial')

# Define column layout
col1, col2 = st.columns([1.5, 1])

# First column with main project description
with col1:
    st.info(
    '''
        ### Project Description
        
        The Flinders trial demonstrates Water Source Australia's Brolga water treatment system in a real-world application.
        This pilot project processes pond water through a multi-barrier treatment approach to achieve potable water quality standards.  
    '''
    )

# Second column with key project details
with col2:
    st.info(
    '''
        ### Key Details  
        
        - **Trial Location**: Flinders Farm, Frankston-Flinders Road  
        - **Source Water**: Farm Pond  
        - **Treatment Goal**: Potable Water Quality  
    '''
    )

# System Overview
st.header('System Overview')
st.markdown("""
    The Brolga treatment system employs multiple barriers for water treatment:
    - Pre-filtration for large particle removal
    - Mixed media filtration for iron and manganese removal
    - Ultrafiltration for pathogen and particle removal
    - Carbon filtration for taste, odor, and color removal
    - UV disinfection for final pathogen inactivation
""")

# System Performance Metrics
st.header('Treatment System Performance')

# Define column layout for performance metrics
col1, col2, col3 = st.columns(3)

# First column for Pathogen Removal details
with col1:
    st.info(
    '''
        ### Pathogen Removal
        
        **✓** >7 log bacteria removal  
        **✓** >6.5 log virus removal  
        **✓** >7 log protozoa removal  
    '''
    )

# Second column for Physical Treatment details
with col2:
    st.info(
    '''
        ### Physical Treatment  
        
        **✓** Turbidity < 0.1 NTU  
        **✓** Color reduction to < 15 HU  
        **✓** TDS reduction to spec  
    '''
    )

# Third column for Chemical Treatment details
with col3:
    st.info(
    '''
        ### Chemical Treatment  
        
        **✓** Iron/Manganese removal  
        **✓** pH correction  
        **✓** Organic carbon reduction  
    '''
    )

# Sidebar
st.sidebar.title('Control Panel')
st.sidebar.markdown('Navigate through the pages to view detailed analysis of:')
st.sidebar.markdown('- Influent Water Analysis')
st.sidebar.markdown('- Treated Water Analysis')
st.sidebar.markdown('- Water Quality Comparison')
st.sidebar.markdown('---')
st.sidebar.warning('Note: Values below detection limits are shown as the detection limit value. Actual values may be lower.')