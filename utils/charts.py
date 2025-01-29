import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st

def get_dynamic_range(data_df, als_lookup, week_cols):
    """Calculate dynamic range for a parameter"""
    param_data = data_df[data_df['ALS Lookup'] == als_lookup]
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

def create_hover_text(param_name, value, min_val, max_val, unit=""):
    """Create hover text for a parameter with unit"""
    if pd.isna(value) or value == 'N/R' or str(value).strip() == '':
        return f"{param_name}: No data available"
        
    try:
        if isinstance(value, str):
            if value.startswith('<'):
                value = float(value.replace('<', ''))
            elif 'LINT' in value:
                value = float(value.split()[0].replace('<', ''))
            else:
                value = float(value)
                
        if str(param_name).upper() == 'PH':
            diff_from_neutral = abs(value - 7.0)
            max_deviation = max(abs(max_val - 7), abs(min_val - 7))
            return (
                f"pH: {value:.2f}<br>" +
                f"Difference from neutral: ±{diff_from_neutral:.2f}<br>" +
                f"Max allowed deviation: ±{max_deviation:.2f}"
            )
        else:
            unit_text = f" {unit}" if unit else ""
            return (
                f"{param_name}<br>" +
                f"Value: {value:.2f}{unit_text}<br>" +
                f"Range: {min_val:.2f} - {max_val:.2f}{unit_text}"
            )
    except (ValueError, TypeError):
        return f"{param_name}: Invalid value"

def create_parameter_table(week_num, als_lookups, data_df, ranges_df):
    """Create a formatted parameter table with units"""
    week_col = f'Week {week_num}'
    
    # Filter data and ranges
    data_filtered = data_df[data_df['ALS Lookup'].isin(als_lookups)].copy()
    ranges_filtered = ranges_df[ranges_df['ALS Lookup'].isin(als_lookups)].copy()
    
    # Create combined display dataframe
    df_display = pd.merge(
        data_filtered[['ALS Lookup', week_col]],
        ranges_filtered[['ALS Lookup', 'Parameter', 'Min', 'Max', 'Unit']],
        on='ALS Lookup',
        how='left'
    )
    
    # Format display
    df_display = df_display.rename(columns={week_col: 'Current Value'})
    
    # Add units to ranges
    df_display['Range'] = df_display.apply(
        lambda x: f"{x['Min']} - {x['Max']}{' ' + x['Unit'] if pd.notna(x['Unit']) else ''}", 
        axis=1
    )
    
    # Add units to current values (except for pH)
    df_display['Current Value'] = df_display.apply(
        lambda x: (f"{x['Current Value']}{' ' + x['Unit'] if pd.notna(x['Unit']) and x['Parameter'] != 'PH' else ''}"
                  if pd.notna(x['Current Value']) else x['Current Value']), 
        axis=1
    )
    
    try:
        # Handle pH difference if present
        ph_params = df_display[df_display['Parameter'].str.upper() == 'PH'] if 'Parameter' in df_display.columns else pd.DataFrame()
        if not ph_params.empty:
            ph_mask = df_display['Parameter'].str.upper() == 'PH'
            df_display.loc[ph_mask, 'pH Difference'] = df_display.loc[ph_mask, 'Current Value'].apply(
                lambda x: abs(float(str(x).split()[0]) - 7.0) if pd.notna(x) and x != 'N/R' else None
            )
            return df_display[['Parameter', 'Current Value', 'pH Difference', 'Range']].set_index('Parameter')
        
        return df_display[['Parameter', 'Current Value', 'Range']].set_index('Parameter')
    except Exception as e:
        print(f"Error in create_parameter_table: {str(e)}")
        # Return a basic dataframe if there's an error
        return pd.DataFrame({
            'Current Value': [],
            'Range': []
        })

def normalize_parameter(value, param_name, min_val, max_val):
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
        min_val = float(min_val) if pd.notna(min_val) else 0
        max_val = float(max_val) if pd.notna(max_val) else 1
        
        if str(param_name).upper() == 'PH':
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

def format_parameter_label(param_name, value, max_val, min_val, unit=""):
    try:
        # Handle string values like '<0.1'
        if isinstance(value, str):
            if value.startswith('<'):
                value = float(value.replace('<', ''))
            elif value == 'N/R':
                return f"{param_name}{unit}"
            elif 'LINT' in value:
                value = float(value.split()[0].replace('<', ''))
            else:
                value = float(value)

        if pd.isna(value) or pd.isna(max_val):
            return f"{param_name}{unit}"

        value = float(value)
        max_val = float(max_val)
        min_val = float(min_val)
        
        unit_text = f" {unit}" if unit else ""
        
        # Determine decimal places based on value
        value_format = '.4f' if value <= 1 else '.1f'
        max_val_format = '.4f' if max_val <= 1 else '.1f'
        
        # Format the base label with current value and max value
        label = f"{param_name} {value:{value_format}} ({min_val:{max_val_format}} - {max_val:{max_val_format}}) {unit_text}"
        
        # Add warning indicator if value exceeds max
        if value > max_val:
            label = "⚠️ " + label
            
        return label
        
    except (ValueError, TypeError):
        return f"{param_name}{unit}"

def create_microbial_display(week_num, als_lookups, data_df, ranges_df):
    """Create a display for microbial parameters showing change from initial values"""
    # Get all week columns
    week_cols = [col for col in data_df.columns if col.startswith('Week')]
    week_cols.sort()  # Ensure weeks are in order
    
    # Filter for microbial parameters
    micro_data = data_df[data_df['ALS Lookup'].isin(als_lookups)].copy()
    micro_ranges = ranges_df[ranges_df['ALS Lookup'].isin(als_lookups)].copy()
    
    # Create display for each parameter
    for _, range_row in micro_ranges.iterrows():
        als_lookup = range_row['ALS Lookup']
        param_name = range_row['Parameter']
        unit = range_row['Unit'] if pd.notna(range_row['Unit']) else ''
        
        param_data = micro_data[micro_data['ALS Lookup'] == als_lookup]
        if not param_data.empty:
            # Get values for all weeks
            values = []
            for week in week_cols:
                try:
                    val = param_data[week].iloc[0]
                    if isinstance(val, str):
                        if val.startswith('<'):
                            val = float(val.replace('<', ''))
                        elif val == 'N/R':
                            val = None
                        else:
                            val = float(val)
                    values.append(val)
                except (ValueError, TypeError, IndexError):
                    values.append(None)
            
            # Create metrics display
            if values[0] is not None:  # If we have an initial value
                initial_value = values[0]
                current_value = values[week_num - 1] if week_num <= len(values) else None
                
                if current_value is not None:
                    reduction = ((initial_value - current_value) / initial_value * 100 
                               if initial_value != 0 else 0)
                    
                    st.metric(
                        label=f"{param_name} ({unit})",
                        value=f"{current_value:,.1f}",
                        delta=f"{reduction:,.1f}% reduction" if reduction > 0 else "No reduction",
                        delta_color="normal" if reduction > 0 else "off"
                    )
                else:
                    st.metric(
                        label=f"{param_name} ({unit})",
                        value="No data",
                        delta=None
                    )
            else:
                st.metric(
                    label=f"{param_name} ({unit})",
                    value="No initial data",
                    delta=None
                )

def create_radar_chart(week_num, als_lookups, data_df, treated_data, ranges_df, treated_ranges, chart_type='comparison', category=None):
    """
    Create a radar chart based on the specified type (influent, treated, or comparison)
    
    Parameters:
    - week_num: Current week number
    - als_lookups: List of ALS lookup values
    - data_df: Main data DataFrame (influent data when chart_type is 'influent' or 'comparison')
    - treated_data: Treated water data DataFrame
    - ranges_df: Main ranges DataFrame
    - treated_ranges: Treated water ranges DataFrame
    - chart_type: Type of chart ('influent', 'treated', or 'comparison')
    - category: Category name for the chart title
    
    Returns:
    - plotly figure object, error message (if any)
    """
    import plotly.graph_objects as go
    import pandas as pd
    
    week_cols = [col for col in data_df.columns if col.startswith('Week')]
    week_col = f'Week {week_num}'
    
    # Filter data and ranges
    ranges_filtered = ranges_df[ranges_df['ALS Lookup'].isin(als_lookups)].copy()
    data_filtered = data_df[data_df['ALS Lookup'].isin(als_lookups)].copy()
    treated_filtered = treated_data[treated_data['ALS Lookup'].isin(als_lookups)].copy()
    treated_ranges_filtered = treated_ranges[treated_ranges['ALS Lookup'].isin(als_lookups)].copy()
    
    def process_parameter_data(param_data, param_ranges):
        """Process data for a single dataset"""
        values = []
        normalized_values = []
        labels = []
        hover_texts = []
        
        for _, range_row in param_ranges.iterrows():
            als_lookup = range_row['ALS Lookup']
            param_name = range_row['Parameter']
            unit = range_row['Unit'] if pd.notna(range_row['Unit']) else ""
            
            # Get parameter data
            row_data = param_data[param_data['ALS Lookup'] == als_lookup]
            if not row_data.empty and week_col in row_data.columns:
                value = row_data[week_col].iloc[0]
                min_val = float(range_row['Min']) if pd.notna(range_row['Min']) else 0
                max_val = float(range_row['Max']) if pd.notna(range_row['Max']) else 1
                
                # Format value with appropriate decimal places
                if value is not None:
                    if isinstance(value, str):
                        if value.startswith('<'):
                            value = float(value.replace('<', ''))
                        elif value == 'N/R':
                            value = None
                        elif 'LINT' in value:
                            value = float(value.split()[0].replace('<', ''))
                        else:
                            value = float(value)
                    
                    if value is not None:
                        value_format = '.4f' if value <= 1 else '.1f'
                        value_text = f"{value:{value_format}}"
                    else:
                        value_text = "N/A"
                else:
                    value_text = "N/A"
                
                # Create range text with unit
                range_text = f"({min_val:.2f} - {max_val:.2f})"
                if unit:
                    range_text += f" {unit}"
                
                # Create label based on chart type and availability of comparison data
                if chart_type == 'comparison' and 'raw_treated' in locals():
                    # Calculate improvement percentage for comparison charts
                    treated_val = raw_treated[len(values)] if raw_treated else None
                    if value is not None and treated_val is not None and value != 0:
                        improvement = ((value - treated_val) / value) * 100
                        label = [
                            f"{param_name}",
                            f"{improvement:.1f}% improvement",
                            f"{value_text} {range_text}"
                        ]
                    else:
                        label = [
                            f"{param_name}",
                            f"{value_text} {range_text}"
                        ]
                else:
                    label = [
                        f"{param_name}",
                        f"{value_text} {range_text}"
                    ]
                
                # Join with HTML line breaks for proper multi-line display
                label_text = '<br>'.join(label)
                labels.append(label_text)
                
                # Process and normalize value
                try:
                    if isinstance(value, str):
                        if value.startswith('<'):
                            value = float(value.replace('<', ''))
                        elif value == 'N/R':
                            value = None
                        elif 'LINT' in value:
                            value = float(value.split()[0].replace('<', ''))
                        else:
                            value = float(value)
                except (ValueError, TypeError):
                    value = None
                
                values.append(value)
                norm_val = normalize_parameter(value, param_name, min_val, max_val)
                normalized_values.append(norm_val)
                
                # Create hover text
                # Create custom hover text
                if value is None:
                    hover_text = f"{param_name}: No data available"
                else:
                    value_format = '.4f' if value <= 1 else '.1f'
                    unit_text = f" {unit}" if unit else ""
                    
                    if chart_type == 'comparison' and 'raw_treated' in locals():
                        treated_val = raw_treated[len(values)] if raw_treated else None
                        if treated_val is not None and value != 0:
                            improvement = ((value - treated_val) / value) * 100
                            hover_text = (
                                f"{param_name}<br>" +
                                f"Improvement: {improvement:.1f}%<br>" +
                                f"Influent: {value:{value_format}}{unit_text}<br>" +
                                f"Treated: {treated_val:{value_format}}{unit_text}<br>" +
                                f"Range: {min_val:.2f} - {max_val:.2f}{unit_text}"
                            )
                        else:
                            hover_text = (
                                f"{param_name}<br>" +
                                f"Value: {value:{value_format}}{unit_text}<br>" +
                                f"Range: {min_val:.2f} - {max_val:.2f}{unit_text}"
                            )
                    else:
                        hover_text = (
                            f"{param_name}<br>" +
                            f"Value: {value:{value_format}}{unit_text}<br>" +
                            f"Range: {min_val:.2f} - {max_val:.2f}{unit_text}"
                        )
                hover_texts.append(hover_text)
        
        return labels, normalized_values, hover_texts, values
    
    # Process both datasets
    influent_labels, influent_values, influent_hovers, raw_influent = process_parameter_data(
        data_filtered, ranges_filtered
    )
    treated_labels, treated_values, treated_hovers, raw_treated = process_parameter_data(
        treated_filtered, treated_ranges_filtered
    )
    
    # Create the figure
    fig = go.Figure()
    
    # Add traces based on chart type
    if chart_type in ['influent', 'comparison']:
        # Add influent trace
        fig.add_trace(go.Scatterpolar(
            r=influent_values,
            theta=influent_labels,
            name='Influent Water',
            fill='toself',
            line=dict(color='#8B4513', shape='spline', smoothing=1.3),
            connectgaps=True,
            hovertemplate="%{text}<br>Quality: %{customdata:.0%}<extra></extra>",
            customdata=[1 - v if v is not None else 0 for v in influent_values],
            text=influent_hovers,
            opacity=0.6
        ))
    
    if chart_type in ['treated', 'comparison']:
        # Add treated trace
        fig.add_trace(go.Scatterpolar(
            r=treated_values,
            theta=treated_labels if chart_type == 'treated' else influent_labels,
            name='Treated Water',
            fill='toself',
            line=dict(color='#1E90FF', shape='spline', smoothing=1.3),
            connectgaps=True,
            hovertemplate="%{text}<br>Quality: %{customdata:.0%}<extra></extra>",
            customdata=[1 - v if v is not None else 0 for v in treated_values],
            text=treated_hovers,
            opacity=0.8
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
                period=len(influent_labels),
                rotation=90,
                tickangle=0,  # Keep text horizontal
                tickmode='array',
                ticktext=influent_labels,
                tickfont=dict(size=10),  # Adjust font size
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True,
        height=500,  # Increased height
        title="",
        margin=dict(t=50, b=50, l=80, r=80),  # Increased margins
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig, None