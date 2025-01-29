import streamlit as st
import pandas as pd
import os
from glob import glob
from functools import lru_cache
from datetime import datetime
import pytz

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

def load_csv_directory(directory_path, pattern="*.csv"):
    # Get list of all matching CSV files
    csv_files = glob(os.path.join(directory_path, pattern))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory_path} matching pattern {pattern}")
    
    # Load each CSV file
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, low_memory=False)
            # Add source file column for debugging if needed
            df['_source_file'] = os.path.basename(file)
            dfs.append(df)
        except Exception as e:
            st.warning(f"Error loading {file}: {str(e)}")
            continue
    
    if not dfs:
        raise ValueError(f"No valid CSV files could be loaded from {directory_path}")
    
    # Concatenate all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Sort by timestamp (assuming 'timestamp' column exists)
    if 'timestamp' in combined_df.columns:
        # Convert timestamp to datetime if it's not already (assuming UTC)
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], dayfirst=False, utc=True)
        
        # Convert UTC to Melbourne time
        melbourne_tz = pytz.timezone('Australia/Melbourne')
        combined_df['timestamp'] = combined_df['timestamp'].dt.tz_convert(melbourne_tz)
        
        # Sort by timestamp
        combined_df = combined_df.sort_values('timestamp')
        # Remove duplicates keeping the latest entry
        combined_df = combined_df.drop_duplicates(
            subset=[col for col in combined_df.columns if col != '_source_file'], 
            keep='last'
        )
    
    # Drop the source file column
    combined_df = combined_df.drop('_source_file', axis=1)
    
    return combined_df

def create_parameter_mapping(params_df):
    """
    Create a mapping between ALS Lookup values and Parameter names.
    Returns a dictionary mapping ALS Lookup codes to their corresponding parameters.
    """
    mapping = {}
    
    # Create mapping for rows with ALS Lookup values
    for _, row in params_df.iterrows():
        als_lookup = str(row['ALS Lookup']).strip()
        if pd.notna(als_lookup) and als_lookup:  # Check if ALS Lookup exists and is not empty
            parameter = row['Parameter']
            mapping[als_lookup] = parameter
            
            # Also add the parameter name as a key mapping to itself
            # This ensures parameters without ALS codes still work
            mapping[parameter.upper()] = parameter
    
    return mapping

def get_parameter_ranges(params_df, water_type='Influent'):
    """
    Extract parameter ranges from the new format.
    
    Args:
        params_df: DataFrame containing parameter information
        water_type: Either 'Influent' or 'Treated' to determine column naming
    
    Returns:
        DataFrame with parameter ranges in the required format
    """
    ranges_data = []
    col_name = f'{water_type} Water'
    
    for _, row in params_df.iterrows():
        parameter = row['Parameter']
        
        # Handle special cases for min/max values
        min_val = 0  # Default minimum
        if pd.notna(row['Min']):
            if str(row['Min']).upper() != 'N/A':
                try:
                    min_val = float(row['Min'])
                except (ValueError, TypeError):
                    pass
        
        max_val = None
        if pd.notna(row['Max']):
            if str(row['Max']).upper() != 'N/A' and 'varies' not in str(row['Max']).lower():
                try:
                    max_val = float(row['Max'])
                except (ValueError, TypeError):
                    pass
        
        # If no max value is specified, use a default
        if max_val is None:
            if parameter.upper() == 'PH':
                max_val = 14
            else:
                max_val = 1000  # Default maximum
        
        ranges_data.append({
            col_name: parameter,
            'Details': f"{parameter} ({row['Unit']})" if pd.notna(row['Unit']) else parameter,
            'Min': min_val,
            'Max': max_val
        })
    
    return pd.DataFrame(ranges_data)

def process_water_data(data_df, parameter_mapping, water_type='Influent'):
    """
    Process water quality data using the parameter mapping.
    
    Args:
        data_df: DataFrame containing water quality data
        parameter_mapping: Dictionary mapping ALS codes to parameter names
        water_type: Either 'Influent' or 'Treated' to determine column naming
    
    Returns:
        Processed DataFrame with standardized parameter names
    """
    processed_df = data_df.copy()
    col_name = f'{water_type} Water' if water_type == 'Influent' else 'Product Water'
    
    # Create a mapping function that handles both ALS codes and direct parameter names
    def map_parameter(param):
        param_upper = str(param).upper()
        return parameter_mapping.get(param_upper, param)
    
    # Apply the mapping to the water quality column
    processed_df[col_name] = processed_df[col_name].apply(map_parameter)
    
    return processed_df

def load_all_data():
    """
    Load all data files with updated parameter processing for both influent and treated water.
    """
    try:
        data = {}
        
        # Load testing CSVs
        data['influent_params'] = pd.read_csv('data/Influent Parameters.csv')
        data['treated_params'] = pd.read_csv('data/Treated Parameters.csv')
        data['influent_data'] = pd.read_csv('data/Influent Water.csv')
        data['treated_data'] = pd.read_csv('data/Treated Water.csv')
        
        # Load remaining data files
        data['info'] = load_csv_directory('data/info', 'Info *.csv')
        data['alarms'] = load_csv_directory('data/alarms', 'Alarms *.csv')
        data['warnings'] = load_csv_directory('data/warnings', 'Warnings *.csv')
        data['sequences'] = load_csv_directory('data/sequences', 'Sequences *.csv')
        data['telemetry'] = load_csv_directory('data/telemetry', 'Telemetry *.csv')
        data['sequence_states'] = pd.read_csv('data/Sequence States.csv')
        data['assets'] = pd.read_csv('data/Assets.csv')
        data['thresholds'] = pd.read_csv('data/Thresholds.csv')
        
        return data
        
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