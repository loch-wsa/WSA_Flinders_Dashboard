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

def create_hover_text(param, value, min_val, max_val):
    """Create hover text for a parameter"""
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
            return (
                f"pH: {value:.2f}<br>" +
                f"Difference from neutral: ±{diff_from_neutral:.2f}<br>" +
                f"Max allowed deviation: ±{max_deviation:.2f}"
            )
        else:
            return (
                f"{param}<br>" +
                f"Value: {value:.2f}<br>" +
                f"Range: {min_val:.2f} - {max_val:.2f}"
            )
    except (ValueError, TypeError):
        return f"{param}: Invalid value"

def create_parameter_table(week_num, params, data_df, ranges_df):
    """Create a formatted parameter table"""
    week_col = f'Week {week_num}'
    
    # Determine the correct column names
    param_col = 'Product Water' if 'Product Water' in data_df.columns else 'Influent Water'
    range_col = 'Treated Water' if 'Treated Water' in ranges_df.columns else 'Influent Water'
    
    # Create combined display dataframe
    df_display = pd.merge(
        data_df[data_df[param_col].isin(params)][['Details', param_col, week_col]],
        ranges_df[[range_col, 'Min', 'Max']],
        left_on=param_col,
        right_on=range_col,
        how='left'
    )
    
    # Format display
    df_display = df_display.rename(columns={week_col: 'Current Value'})
    df_display['Range'] = df_display.apply(lambda x: f"{x['Min']} - {x['Max']}", axis=1)
    
    # Handle pH difference if present
    if 'PH' in params:
        mask = df_display[param_col] == 'PH'
        df_display.loc[mask, 'pH Difference'] = abs(
            df_display.loc[mask, 'Current Value'].astype(float) - 7.0
        )
        return df_display[['Details', 'Current Value', 'pH Difference', 'Range']].set_index('Details')
    
    return df_display[['Details', 'Current Value', 'Range']].set_index('Details')
    
def normalize_parameter(value, param, min_val, max_val):
    """
    Normalize parameter values with special handling for pH
    """
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

def format_parameter_label(param, value, max_val, comparison_value=None, is_treated=False, show_comparison=False):
    """
    Format parameter label with status indicators and appropriate ratios:
    - For comparison view: treated/raw ratio
    - For single view: value/max ratio
    Status indicators:
    - Green checkmark (✅) for values within limits
    - Warning (⚠️) for values exceeding limits
    - Checkmark (✅) for improved treated water values
    """
    try:
        if isinstance(value, str):
            if value.startswith('<'):
                value = float(value.replace('<', ''))
            elif value == 'N/R':
                return param
            elif 'LINT' in value:
                value = float(value.split()[0].replace('<', ''))
            else:
                value = float(value)

        if pd.isna(value) or pd.isna(max_val):
            return param

        value = float(value)
        max_val = float(max_val)
        
        # Handle comparison mode
        if show_comparison and comparison_value is not None:
            try:
                comparison_value = float(comparison_value)
                # Format ratio as treated/raw
                if is_treated:
                    ratio = f"{value:.1f}/{comparison_value:.1f}"
                else:
                    ratio = f"{comparison_value:.1f}/{value:.1f}"
                
                # Special handling for pH in comparison mode
                if str(param).upper() == 'PH':
                    if value > max_val:
                        return f"⚠️ {param} ({ratio})"
                    is_improved = abs(value - 7.0) < abs(comparison_value - 7.0)
                    if is_improved and is_treated:
                        return f"{param} ({ratio})"
                else:
                    # For other parameters in comparison mode
                    if is_treated:
                        is_improved = value < comparison_value
                        if is_improved:
                            return f"{param} ({ratio})"
                
                # Default comparison display
                if value > max_val:
                    return f"⚠️ {param} ({ratio})"
                return f"{param} ({ratio})"
                
            except (ValueError, TypeError):
                return param
        
        # Non-comparison mode: show value/max ratio
        ratio = f"{value:.1f}/{max_val:.1f}"
        if value > max_val:
            return f"⚠️ {param} ({ratio})"
        return f"{param} ({ratio})"
        
    except (ValueError, TypeError):
        return param

def create_radar_chart(week_num, params, influent_data, treated_data, influent_ranges, treated_ranges, data_type='influent', show_comparison=False, use_brolga_limits=True):
    """Create a radar chart for water quality parameters"""
    week_cols = [col for col in influent_data.columns if col.startswith('Week')]
    
    # Select appropriate dataframes based on data type
    primary_df = treated_data if data_type == 'treated' else influent_data
    primary_ranges = treated_ranges if data_type == 'treated' else influent_ranges
    
    # Determine column names
    if data_type == 'treated':
        param_col = 'Product Water'
        range_param_key = 'Treated Water'
    else:
        param_col = 'Influent Water'
        range_param_key = 'Influent Water'
    
    # Filter data and sort by parameter
    df_filtered = primary_df[primary_df[param_col].isin(params)].copy()
    df_filtered = df_filtered.sort_values(param_col)
    week_col = f'Week {week_num}'
    
    # Get parameter names and values
    param_names = df_filtered[param_col].tolist()
    values = df_filtered[week_col].tolist()
    normalized_values = []
    hover_texts = []
    formatted_param_names = []
    
    # Process each parameter
    for param, value in zip(param_names, values):
        if show_comparison:
            # For comparison charts, always use dynamic ranges
            min_val, max_val = get_dynamic_range(primary_df, param_col, param, week_cols)
        else:
            if use_brolga_limits:
                # Use Brolga limits
                range_row = primary_ranges[primary_ranges[range_param_key] == param]
                min_val = float(range_row['Min'].iloc[0]) if not range_row.empty else 0
                max_val = float(range_row['Max'].iloc[0]) if not range_row.empty else 1
            else:
                # Use dynamic ranges
                min_val, max_val = get_dynamic_range(primary_df, param_col, param, week_cols)
        
        # Get comparison value if in comparison mode
        comparison_value = None
        if show_comparison:
            comparison_df = treated_data if data_type == 'influent' else influent_data
            comparison_param_col = 'Product Water' if data_type == 'influent' else 'Influent Water'
            comparison_filtered = comparison_df[comparison_df[comparison_param_col] == param]
            if not comparison_filtered.empty:
                comparison_value = comparison_filtered[week_col].iloc[0]
        
        # Format parameter name with appropriate indicators and ratio format
        formatted_name = format_parameter_label(
            param, 
            value, 
            max_val,
            comparison_value=comparison_value,
            is_treated=(data_type == 'treated'),
            show_comparison=show_comparison
        )
        formatted_param_names.append(formatted_name)
        
        # Normalize value
        norm_val = normalize_parameter(value, param, min_val, max_val)
        normalized_values.append(norm_val)
        hover_texts.append(create_hover_text(param, value, min_val, max_val))
    
    # Close the shapes
    formatted_param_names.append(formatted_param_names[0])
    normalized_values.append(normalized_values[0])
    hover_texts.append(hover_texts[0])
    
    # Create figure
    fig = go.Figure()
    
    # Add main trace
    primary_color = '#1E90FF' if data_type == 'treated' else '#8B4513'
    fig.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=formatted_param_names,  # Use formatted names with warnings
        name='Treated Water' if data_type == 'treated' else 'Influent Water',
        fill='toself',
        line=dict(color=primary_color, shape='linear'),
        connectgaps=True,
        hovertemplate="%{text}<br>Quality: %{customdata:.0%}<extra></extra>",
        customdata=[1 - v for v in normalized_values],
        text=hover_texts
    ))
    
    # Add comparison trace if requested
    if show_comparison:
        comparison_df = treated_data if data_type == 'influent' else influent_data
        comparison_param_col = 'Product Water' if data_type == 'influent' else 'Influent Water'
        
        comp_filtered = comparison_df[comparison_df[comparison_param_col].isin(params)].sort_values(comparison_param_col)
        comparison_values = comp_filtered[week_col].tolist()
        comparison_normalized = []
        comparison_hover = []
        
        for param, value in zip(param_names[:-1], comparison_values):
            min_val, max_val = get_dynamic_range(comparison_df, comparison_param_col, param, week_cols)
            norm_val = normalize_parameter(value, param, min_val, max_val)
            comparison_normalized.append(norm_val)
            comparison_hover.append(create_hover_text(param, value, min_val, max_val))
        
        # Close the shapes
        comparison_normalized.append(comparison_normalized[0])
        comparison_hover.append(comparison_hover[0])
        
        comparison_color = '#1E90FF' if data_type == 'influent' else '#8B4513'
        fig.add_trace(go.Scatterpolar(
            r=comparison_normalized,
            theta=formatted_param_names,  # Use formatted names with warnings
            name='Treated Water' if data_type == 'influent' else 'Influent Water',
            fill='toself',
            line=dict(color=comparison_color, shape='linear'),
            connectgaps=True,
            hovertemplate="%{text}<br>Quality: %{customdata:.0%}<extra></extra>",
            customdata=[1 - v for v in comparison_normalized],
            text=comparison_hover
        ))
    
    # Update layout (rest of the layout code remains the same)
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
                period=len(formatted_param_names) - 1,
                rotation=90
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=False,
        height=600,
        title=dict(
            text=f"Water Quality Parameters - Week {week_num}",
            x=0.5,
            y=0.95,
            xanchor='center'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Check for missing values
    missing_params = []
    for param, value in zip(param_names[:-1], values[:-1]):
        if pd.isna(value) or value == 'N/R' or str(value).strip() == '':
            missing_params.append(param)
    
    warning_message = None
    if missing_params:
        warning_message = f"Warning: Missing values for Week {week_num} in parameters: {', '.join(missing_params)}"
    
    return fig, warning_message
