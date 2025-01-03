import streamlit as st
import pandas as pd
import numpy as np
from functools import lru_cache

@lru_cache(maxsize=None)
def process_data(value):
    """Process data values with caching for better performance"""
    try:
        if isinstance(value, str):
            if value.startswith('<'):
                return float(value.strip('<'))
            elif value.startswith('>'):
                value = value.strip('>')
                if value.isdigit():
                    return float(value)
                return 2000
            elif value == 'N/R':
                return 0
            elif value.endswith('LINT'):
                return float(value.split()[0].strip('<'))
            try:
                return float(value)
            except ValueError:
                return 0
        elif value is None:
            return 0
        else:
            return float(value)
    except Exception:
        return 0

def normalize_parameter(value, param, min_val, max_val):
    """Normalize parameter values with special handling for pH"""
    param_str = str(param).upper()
    
    if param_str == 'PH':
        try:
            value = float(value)
            diff_from_neutral = abs(value - 7.0)
            max_deviation = max(abs(max_val - 7), abs(min_val - 7))
            return diff_from_neutral / max_deviation if max_deviation != 0 else 0
        except (ValueError, TypeError):
            return 0
    else:
        try:
            value = float(value)
            range_size = max_val - min_val
            return (value - min_val) / range_size if range_size != 0 else 0
        except (ValueError, TypeError):
            return 0

def calculate_dynamic_ranges(data_df, param_col):
    """Calculate dynamic ranges based on actual data"""
    week_cols = [col for col in data_df.columns if col.startswith('Week')]
    ranges_data = []
    
    # Map the parameter column names to match Brolga format
    output_col_name = 'Treated Water' if param_col == 'Product Water' else 'Influent Water'
    
    for param in data_df[param_col].unique():
        param_data = data_df[data_df[param_col] == param]
        values = []
        
        for col in week_cols:
            try:
                val = param_data[col].iloc[0]
                if pd.notna(val) and val != 'N/R':
                    processed_val = process_data(val)
                    if processed_val > 0:
                        values.append(processed_val)
            except (IndexError, ValueError):
                continue
        
        if values:
            max_val = max(values) * 1.1
            ranges_data.append({
                output_col_name: param,  # Use the mapped column name
                'Details': param_data['Details'].iloc[0] if not param_data['Details'].empty else param,
                'Min': 0,
                'Max': max_val
            })
        else:
            ranges_data.append({
                output_col_name: param,  # Use the mapped column name
                'Details': param_data['Details'].iloc[0] if not param_data['Details'].empty else param,
                'Min': 0,
                'Max': 1
            })
            
    return pd.DataFrame(ranges_data)

def prepare_chart_data(df, ranges_df, param_col, week_num):
    """Prepare normalized data for charting"""
    week_col = f'Week {week_num}'
    chart_data = []
    
    for _, row in df.iterrows():
        param = row[param_col]
        value = row[week_col]
        
        # Get range values
        range_row = ranges_df[ranges_df[param_col] == param]
        if range_row.empty:
            min_val, max_val = 0, 1
        else:
            min_val = float(range_row['Min'].iloc[0])
            max_val = float(range_row['Max'].iloc[0])
        
        # Process value
        actual_val = process_data(value)
        norm_val = normalize_parameter(actual_val, param, min_val, max_val)
        
        # Create hover text
        if str(param).upper() == 'PH':
            if actual_val == 0:
                hover_text = f"pH: No data available"
            else:
                diff_from_neutral = abs(actual_val - 7.0)
                hover_text = f"pH: {actual_val:.2f}\nDifference from neutral: ±{diff_from_neutral:.2f}"
        else:
            hover_text = f"{param}: {actual_val:.2f}\nRange: {min_val:.2f} - {max_val:.2f}"
        
        chart_data.append({
            'parameter': param,
            'actual_value': actual_val,
            'normalized_value': norm_val,
            'hover_text': hover_text,
            'min': min_val,
            'max': max_val
        })
    
    return pd.DataFrame(chart_data)

@st.cache_data(ttl=3600)
def load_data(use_brolga_limits=True):
    """Load and process all data files with caching"""
    try:
        # Load raw data
        influent_data = pd.read_csv('data/Point Leo Influent Water.csv')
        treated_data = pd.read_csv('data/Point Leo Treated Water.csv')
        influent_ranges = pd.read_csv('data/Brolga Influent Parameters.csv')
        treated_ranges = pd.read_csv('data/Brolga Treated Parameters.csv')
        
        # Process numeric columns
        for df in [influent_data, treated_data]:
            for col in df.columns:
                if col not in ['Influent Water', 'Product Water', 'Details', 'Pond']:
                    df[col] = df[col].apply(process_data)
        
        # If not using Brolga limits, calculate dynamic ranges
        if not use_brolga_limits:
            influent_ranges = calculate_dynamic_ranges(influent_data, 'Influent Water')
            treated_ranges = calculate_dynamic_ranges(treated_data, 'Product Water')
        
        return influent_data, treated_data, influent_ranges, treated_ranges
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        raise e

def get_chart_data(week_num, params, data_type='influent', show_comparison=False):
    """Get processed data ready for charting"""
    # Load data
    influent_data, treated_data, influent_ranges, treated_ranges = load_data()
    
    # Filter and prepare primary data
    primary_df = treated_data if data_type == 'treated' else influent_data
    primary_ranges = treated_ranges if data_type == 'treated' else influent_ranges
    param_col = 'Product Water' if data_type == 'treated' else 'Influent Water'
    
    filtered_data = primary_df[primary_df[param_col].isin(params)].copy()
    primary_chart_data = prepare_chart_data(filtered_data, primary_ranges, param_col, week_num)
    
    result = {'primary': primary_chart_data}
    
    # Prepare comparison data if needed
    if show_comparison:
        comparison_df = influent_data if data_type == 'treated' else treated_data
        comparison_ranges = influent_ranges if data_type == 'treated' else treated_ranges
        comparison_param_col = 'Influent Water' if data_type == 'treated' else 'Product Water'
        
        filtered_comparison = comparison_df[comparison_df[comparison_param_col].isin(params)].copy()
        comparison_chart_data = prepare_chart_data(filtered_comparison, comparison_ranges, comparison_param_col, week_num)
        result['comparison'] = comparison_chart_data
    
    return result

# Define relevant parameters
RELEVANT_PARAMS = [
    'TURBIDITY',
    'PH',
    'TOC',
    'E COLI(C)',
    'EC',
    'TDS_180',
    'COLIFORM (C)',
    'DOC'
]