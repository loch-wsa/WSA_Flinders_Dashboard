import plotly.graph_objects as go
import pandas as pd
import numpy as np

def get_dynamic_range(data_df, param_col, param, week_cols):
    """Calculate dynamic range for a parameter"""
    param_data = data_df[data_df[param_col] == param]
    values = []
    
    for col in week_cols:
        try:
            val = param_data[col].iloc[0]
            if pd.notna(val) and val != 'N/R':
                if isinstance(val, str):
                    if val.startswith('<'):
                        val = float(val.replace('<', ''))
                    elif 'LINT' in val:
                        val = float(val.split()[0].replace('<', ''))
                    else:
                        val = float(val)
                if val > 0:
                    values.append(val)
        except (IndexError, ValueError):
            continue
    
    if values:
        max_val = max(values) * 1.1  # 10% greater than the worst data point
        return 0, max_val
    return 0, 1

def normalize_parameter(value, param, min_val, max_val):
    """Normalize parameter values with special handling for pH"""
    try:
        # Convert string values like '<0.1' to floats
        if isinstance(value, str):
            if value.startswith('<'):
                value = float(value.replace('<', ''))
            elif value == 'N/R':
                return 0
            elif 'LINT' in value:
                value = float(value.split()[0].replace('<', ''))
            else:
                value = float(value)
                
        value = float(value) if value is not None else 0
        min_val = float(min_val)
        max_val = float(max_val)
        
        if str(param).upper() == 'PH':
            # For pH, calculate difference from neutral (7)
            diff_from_neutral = abs(value - 7.0)
            max_deviation = max(abs(max_val - 7), abs(min_val - 7))
            return diff_from_neutral / max_deviation if max_deviation != 0 else 0
        else:
            # For all other parameters, use standard normalization
            value_range = max_val - min_val
            if value_range == 0:
                return 0
            return (value - min_val) / value_range
            
    except (ValueError, TypeError):
        return 0

def create_hover_text(param, value, min_val, max_val, unit="", notes=""):
    """Create hover text for a parameter with unit and notes"""
    if pd.isna(value) or value == 'N/R' or str(value).strip() == '':
        return f"{param}: No data available"
        
    try:
        if isinstance(value, str):
            if value.startswith('<'):
                value = float(value.replace('<', ''))
            elif 'LINT' in value:
                value = float(value.split()[0].replace('<', ''))
            else:
                value = float(value)
                
        if str(param).upper() == 'PH':
            diff_from_neutral = abs(value - 7.0)
            max_deviation = max(abs(max_val - 7), abs(min_val - 7))
            hover_text = (
                f"pH: {value:.2f}<br>" +
                f"Difference from neutral: ±{diff_from_neutral:.2f}<br>" +
                f"Max allowed deviation: ±{max_deviation:.2f}"
            )
        else:
            unit_text = f" {unit}" if unit else ""
            hover_text = (
                f"{param}<br>" +
                f"Value: {value:.2f}{unit_text}<br>" +
                f"Range: {min_val:.2f} - {max_val:.2f}{unit_text}"
            )
        
        # Add notes if available
        if notes:
            hover_text += f"<br>Note: {notes}"
            
        return hover_text
        
    except (ValueError, TypeError):
        return f"{param}: Invalid value"

def format_parameter_label(param, value, max_val, unit="", category="", comparison_value=None, is_treated=False, show_comparison=False):
    """Format parameter label with units and status indicators"""
    try:
        if isinstance(value, str):
            if value.startswith('<'):
                value = float(value.replace('<', ''))
            elif value == 'N/R':
                return f"{param}{unit}"
            elif 'LINT' in value:
                value = float(value.split()[0].replace('<', ''))
            else:
                value = float(value)

        if pd.isna(value) or pd.isna(max_val):
            return f"{param}{unit}"

        value = float(value)
        max_val = float(max_val)
        unit_text = f" {unit}" if unit else ""
        
        # Handle comparison mode
        if show_comparison and comparison_value is not None:
            try:
                comparison_value = float(comparison_value)
                ratio = f"{value:.1f}/{comparison_value:.1f} {unit_text}"
                
                if str(param).upper() == 'PH':
                    if value > max_val:
                        return f"⚠️ {param} {ratio}"
                    is_improved = abs(value - 7.0) < abs(comparison_value - 7.0)
                    if is_improved and is_treated:
                        return f"{param} {ratio}"
                else:
                    if is_treated:
                        is_improved = value < comparison_value
                        if is_improved:
                            return f"{param} {ratio}"
                
                if value > max_val:
                    return f"⚠️ {param} {ratio}"
                return f"{param} {ratio}"
                
            except (ValueError, TypeError):
                return f"{param}{unit}"
        
        # Non-comparison mode
        ratio = f"{value:.1f}/{max_val:.1f} {unit_text}"
        if value > max_val:
            return f"⚠️ {param} {ratio}"
        return f"{param} {ratio}"
        
    except (ValueError, TypeError):
        return f"{param}{unit}"
 
def create_parameter_table(week_num, params, data_df, params_df):
    """Create a formatted parameter table with units"""
    week_col = f'Week {week_num}'
    
    # Determine which type of data we're dealing with
    data_col = 'Product Water' if 'Product Water' in data_df.columns else 'Influent Water'
    
    # Create display dataframe
    df_display = []
    
    for param in params:
        if param in data_df[data_col].values:
            param_data = data_df[data_df[data_col] == param]
            param_info = params_df[params_df['Parameter'] == param]
            
            if not param_data.empty and not param_info.empty:
                value = param_data[week_col].iloc[0]
                unit = param_info['Unit'].iloc[0] if pd.notna(param_info['Unit'].iloc[0]) else ''
                min_val = param_info['Min'].iloc[0] if pd.notna(param_info['Min'].iloc[0]) else '0'
                max_val = param_info['Max'].iloc[0] if pd.notna(param_info['Max'].iloc[0]) else '1000'
                
                df_display.append({
                    'Parameter': param,
                    'Current Value': f"{value}{' ' + unit if unit and param.upper() != 'PH' else ''}",
                    'Range': f"{min_val} - {max_val}{' ' + unit if unit else ''}"
                })
    
    df_display = pd.DataFrame(df_display)
    
    # Add pH difference if present
    if 'PH' in params:
        mask = df_display['Parameter'] == 'PH'
        df_display.loc[mask, 'pH Difference'] = df_display.loc[mask, 'Current Value'].apply(
            lambda x: abs(float(str(x).split()[0]) - 7.0) if pd.notna(x) and x != 'N/R' else None
        )
    
    return df_display.set_index('Parameter')

def map_parameters_to_ranges(water_data, params_df, data_type='Influent'):
    """Map water quality parameters to their ranges"""
    param_ranges = []
    
    # Determine which type of water we're dealing with
    is_treated = data_type == 'Treated'
    param_col = 'Product Water' if is_treated else 'Influent Water'
    
    for _, row in params_df.iterrows():
        parameter = row['Parameter']
        category = row['Category']
        
        if parameter in water_data[param_col].values:
            # Get min/max values
            min_val = 0  # Default minimum
            if pd.notna(row['Min']) and row['Min'] != 'N/A':
                try:
                    min_val = float(row['Min'])
                except (ValueError, TypeError):
                    pass
            
            max_val = None
            if pd.notna(row['Max']) and row['Max'] != 'N/A':
                try:
                    max_val = float(row['Max'])
                except (ValueError, TypeError):
                    max_val = 1000  # Default maximum if conversion fails
            
            if max_val is None:
                if parameter.upper() == 'PH':
                    max_val = 14
                else:
                    max_val = 1000  # Default maximum
            
            param_ranges.append({
                param_col: parameter,
                'Parameter': parameter,
                'Category': category,
                'Unit': row['Unit'] if pd.notna(row['Unit']) else '',
                'Min': min_val,
                'Max': max_val,
                'Notes': row['Notes'] if pd.notna(row['Notes']) else ''
            })
    
    return pd.DataFrame(param_ranges)

def create_radar_chart(week_num, influent_data, treated_data, influent_params, treated_params, data_type='influent', show_comparison=False, use_brolga_limits=True):
    """Create a radar chart for water quality parameters"""
    # Get the appropriate data based on type
    if data_type == 'treated':
        primary_df = treated_data
        primary_params = treated_params
        data_col = 'Product Water'
    else:
        primary_df = influent_data
        primary_params = influent_params
        data_col = 'Influent Water'
    
    # Filter out microbial parameters
    non_microbial_params = primary_params[
        primary_params['Category'].str.upper() != 'MICROBIAL'
    ]['Parameter'].unique()
    
    # Get parameters present in both the data and params
    available_params = []
    for param in non_microbial_params:
        if param in primary_df[data_col].values:
            available_params.append(param)
    
    df_filtered = primary_df[primary_df[data_col].isin(available_params)].copy()
    
    week_col = f'Week {week_num}'
    param_names = df_filtered[data_col].tolist()
    values = df_filtered[week_col].tolist()
    
    # Process values and create chart data
    normalized_values = []
    hover_texts = []
    formatted_labels = []
    
    for param, value in zip(param_names, values):
        # Get parameter info from params dataframe
        param_info = primary_params[primary_params['Parameter'] == param]
        if param_info.empty:
            continue
        
        # Get min/max values
        min_val = 0
        if pd.notna(param_info['Min'].iloc[0]) and param_info['Min'].iloc[0] != 'N/A':
            try:
                min_val = float(param_info['Min'].iloc[0])
            except (ValueError, TypeError):
                pass
        
        max_val = 1000  # Default maximum
        if pd.notna(param_info['Max'].iloc[0]) and param_info['Max'].iloc[0] != 'N/A':
            try:
                max_val = float(param_info['Max'].iloc[0])
            except (ValueError, TypeError):
                pass
        
        # Special handling for pH
        if str(param).upper() == 'PH':
            max_val = 14
        
        # Normalize and format
        norm_val = normalize_parameter(value, param, min_val, max_val)
        normalized_values.append(norm_val)
        
        # Create hover text and labels
        unit = param_info['Unit'].iloc[0] if pd.notna(param_info['Unit'].iloc[0]) else ''
        notes = param_info['Notes'].iloc[0] if pd.notna(param_info['Notes'].iloc[0]) else ''
        
        hover_text = create_hover_text(param, value, min_val, max_val, unit, notes)
        hover_texts.append(hover_text)
        
        label = format_parameter_label(param, value, max_val, unit)
        formatted_labels.append(label)
    
    # Close the radar chart shapes
    if formatted_labels:  # Only if we have data
        formatted_labels.append(formatted_labels[0])
        normalized_values.append(normalized_values[0])
        hover_texts.append(hover_texts[0])
    
    # Create the plot
    fig = go.Figure()
    
    if normalized_values:  # Only create traces if we have data
        # Add main trace
        primary_color = '#1E90FF' if data_type == 'treated' else '#8B4513'
        fig.add_trace(go.Scatterpolar(
            r=normalized_values,
            theta=formatted_labels,
            name='Treated Water' if data_type == 'treated' else 'Influent Water',
            fill='toself',
            line=dict(color=primary_color),
            hovertemplate="%{text}<br>Quality: %{customdata:.0%}<extra></extra>",
            customdata=[1 - v for v in normalized_values],
            text=hover_texts
        ))
        
        # Add comparison trace if requested
        if show_comparison:
            comparison_df = treated_data if data_type == 'influent' else influent_data
            comparison_params = treated_params if data_type == 'influent' else influent_params
            comparison_col = 'Product Water' if data_type == 'influent' else 'Influent Water'
            
            comp_values = []
            comp_hover = []
            
            for param in param_names[:-1]:  # Exclude the closing point
                comp_row = comparison_df[comparison_df[comparison_col] == param]
                if not comp_row.empty:
                    value = comp_row[week_col].iloc[0]
                    param_info = comparison_params[comparison_params['Parameter'] == param].iloc[0]
                    
                    min_val = 0
                    max_val = 1000
                    if pd.notna(param_info['Min']) and param_info['Min'] != 'N/A':
                        try:
                            min_val = float(param_info['Min'])
                        except (ValueError, TypeError):
                            pass
                    if pd.notna(param_info['Max']) and param_info['Max'] != 'N/A':
                        try:
                            max_val = float(param_info['Max'])
                        except (ValueError, TypeError):
                            pass
                    
                    norm_val = normalize_parameter(value, param, min_val, max_val)
                    comp_values.append(norm_val)
                    
                    unit = param_info['Unit'] if pd.notna(param_info['Unit']) else ''
                    notes = param_info['Notes'] if pd.notna(param_info['Notes']) else ''
                    hover_text = create_hover_text(param, value, min_val, max_val, unit, notes)
                    comp_hover.append(hover_text)
            
            if comp_values:  # Only add comparison trace if we have data
                comp_values.append(comp_values[0])
                comp_hover.append(comp_hover[0])
                
                comparison_color = '#1E90FF' if data_type == 'influent' else '#8B4513'
                fig.add_trace(go.Scatterpolar(
                    r=comp_values,
                    theta=formatted_labels,
                    name='Treated Water' if data_type == 'influent' else 'Influent Water',
                    fill='toself',
                    line=dict(color=comparison_color),
                    hovertemplate="%{text}<br>Quality: %{customdata:.0%}<extra></extra>",
                    customdata=[1 - v for v in comp_values],
                    text=comp_hover
                ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickmode='array',
                ticktext=['Ideal', 'Good', 'Fair', 'Poor', 'Critical'],
                tickvals=[0, 0.25, 0.5, 0.75, 1]
            ),
            angularaxis=dict(
                direction="clockwise",
                period=len(formatted_labels) - 1 if formatted_labels else 1
            )
        ),
        showlegend=show_comparison,
        height=600,
        title=f"Water Quality Parameters - Week {week_num}"
    )
    
    warning = None
    if not normalized_values:
        warning = "No non-microbial parameters found for the selected week"
    
    return fig, warning
