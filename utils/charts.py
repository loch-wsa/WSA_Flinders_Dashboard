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

def format_parameter_label(param_name, value, max_val, unit=""):
    """Format parameter label with units and status indicators"""
    try:
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
        unit_text = f" {unit}" if unit else ""
        
        # Format the base label with value and unit
        label = f"{param_name} ({value:.1f}{unit_text})"
        
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

def create_category_radar_chart(week_num, als_lookups, data_df, treated_data, ranges_df, treated_ranges, data_type='influent', category=None):
    """Create a radar chart for a specific category of parameters"""
    week_cols = [col for col in data_df.columns if col.startswith('Week')]
    week_col = f'Week {week_num}'
    
    # Filter data and ranges
    ranges_filtered = ranges_df[ranges_df['ALS Lookup'].isin(als_lookups)].copy()
    data_filtered = data_df[data_df['ALS Lookup'].isin(als_lookups)].copy()
    
    # Get values and prepare chart data
    values = []
    normalized_values = []
    hover_texts = []
    formatted_param_names = []
    
    for _, range_row in ranges_filtered.iterrows():
        als_lookup = range_row['ALS Lookup']
        param_name = range_row['Parameter']
        
        # Get parameter data
        param_data = data_filtered[data_filtered['ALS Lookup'] == als_lookup]
        
        if not param_data.empty and week_col in param_data.columns:
            value = param_data[week_col].iloc[0]
            min_val = float(range_row['Min']) if not pd.isna(range_row['Min']) else 0
            max_val = float(range_row['Max']) if not pd.isna(range_row['Max']) else 1
            unit = range_row['Unit'] if 'Unit' in range_row and pd.notna(range_row['Unit']) else ""
            
            # Format parameter name and normalize value
            formatted_name = format_parameter_label(param_name, value, max_val, unit)
            formatted_param_names.append(formatted_name)
            
            norm_val = normalize_parameter(value, param_name, min_val, max_val)
            normalized_values.append(norm_val)
            
            hover_text = create_hover_text(param_name, value, min_val, max_val, unit)
            hover_texts.append(hover_text)
    
    # Handle shape closure based on number of parameters
    if len(formatted_param_names) == 1:
        # For single parameter, create a circle by adding more points
        val = normalized_values[0]
        name = formatted_param_names[0]
        hover = hover_texts[0]
        
        # Add additional points to create a smooth circle
        angles = [0, 60, 120, 180, 240, 300, 0]  # Seven points for a smooth circle
        for angle in angles:
            normalized_values.append(val)
            formatted_param_names.append(name)
            hover_texts.append(hover)
    elif len(formatted_param_names) == 2:
        # For two parameters, add intermediate points for smoother curve
        val1, val2 = normalized_values
        name1, name2 = formatted_param_names
        hover1, hover2 = hover_texts
        
        # Add intermediate points
        normalized_values = [val1, val1, val2, val2, val1]
        formatted_param_names = [name1, name1, name2, name2, name1]
        hover_texts = [hover1, hover1, hover2, hover2, hover1]
    else:
        # For more than two parameters, just close the shape
        formatted_param_names.append(formatted_param_names[0])
        normalized_values.append(normalized_values[0])
        hover_texts.append(hover_texts[0])
    
    # Create figure
    fig = go.Figure()
    
    if normalized_values:
        # Base color for influent/treated water
        primary_color = '#1E90FF' if data_type == 'treated' else '#8B4513'
        
        fig.add_trace(go.Scatterpolar(
            r=normalized_values,
            theta=formatted_param_names,
            name=category if category else ('Treated Water' if data_type == 'treated' else 'Influent Water'),
            fill='toself',
            line=dict(
                color=primary_color,
                shape='spline',  # Use spline for smooth curves
                smoothing=1.3    # Adjust smoothing factor for more circular paths
            ),
            connectgaps=True,
            hovertemplate="%{text}<br>Quality: %{customdata:.0%}<extra></extra>",
            customdata=[1 - v for v in normalized_values],
            text=hover_texts
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
                period=len(formatted_param_names) if formatted_param_names else 1,
                rotation=90
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=False,
        height=400,  # Smaller height for the category charts
        title=None,  # Remove title as it's shown in the subheader
        margin=dict(t=20, b=20, l=40, r=40),  # Reduce margins
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig, None
    week_cols = [col for col in data_df.columns if col.startswith('Week')]
    week_col = f'Week {week_num}'
    
    # Filter data and ranges
    ranges_filtered = ranges_df[ranges_df['ALS Lookup'].isin(als_lookups)].copy()
    data_filtered = data_df[data_df['ALS Lookup'].isin(als_lookups)].copy()
    
    # Get values and prepare chart data
    values = []
    normalized_values = []
    hover_texts = []
    formatted_param_names = []
    
    for _, range_row in ranges_filtered.iterrows():
        als_lookup = range_row['ALS Lookup']
        param_name = range_row['Parameter']
        
        # Get parameter data
        param_data = data_filtered[data_filtered['ALS Lookup'] == als_lookup]
        
        if not param_data.empty and week_col in param_data.columns:
            value = param_data[week_col].iloc[0]
            min_val = float(range_row['Min']) if not pd.isna(range_row['Min']) else 0
            max_val = float(range_row['Max']) if not pd.isna(range_row['Max']) else 1
            unit = range_row['Unit'] if 'Unit' in range_row and pd.notna(range_row['Unit']) else ""
            
            # Format parameter name and normalize value
            formatted_name = format_parameter_label(param_name, value, max_val, unit)
            formatted_param_names.append(formatted_name)
            
            norm_val = normalize_parameter(value, param_name, min_val, max_val)
            normalized_values.append(norm_val)
            
            hover_text = create_hover_text(param_name, value, min_val, max_val, unit)
            hover_texts.append(hover_text)
    
    # Close the shapes if we have data
    if formatted_param_names:
        formatted_param_names.append(formatted_param_names[0])
        normalized_values.append(normalized_values[0])
        hover_texts.append(hover_texts[0])
    
    # Create figure
    fig = go.Figure()
    
    if normalized_values:
        # Base color for influent/treated water
        primary_color = '#1E90FF' if data_type == 'treated' else '#8B4513'
        
        # Group parameters by category
        data_by_category = {}
        for i, (_, range_row) in enumerate(ranges_filtered.iterrows()):
            category = range_row['Category']
            if category not in data_by_category:
                data_by_category[category] = {
                    'normalized': [],
                    'params': [],
                    'hover': []
                }
            if i < len(normalized_values)-1:  # Exclude the closing point
                data_by_category[category]['normalized'].append(normalized_values[i])
                data_by_category[category]['params'].append(formatted_param_names[i])
                data_by_category[category]['hover'].append(hover_texts[i])
        
        # Define line styles for different categories
        category_styles = {
            'Physical': dict(dash='solid'),
            'Inorganic Compound': dict(dash='dot'),
            'Organic Compound': dict(dash='dash'),
            'Metal': dict(dash='longdash'),
            'Radiological': dict(dash='dashdot'),
            'Algae Toxins': dict(dash='longdashdot')
        }
        
        # Add traces for each category
        for category, cat_data in data_by_category.items():
            if cat_data['normalized']:  # Only add trace if we have data
                # Add first value at end to close the shape
                cat_normalized = cat_data['normalized'] + [cat_data['normalized'][0]]
                cat_params = cat_data['params'] + [cat_data['params'][0]]
                cat_hover = cat_data['hover'] + [cat_data['hover'][0]]
                
                # Get line style for this category
                line_style = category_styles.get(category, dict(dash='solid'))
                
                fig.add_trace(go.Scatterpolar(
                    r=cat_normalized,
                    theta=cat_params,
                    name=category,
                    line=dict(
                        color=primary_color,
                        shape='linear',
                        **line_style
                    ),
                    connectgaps=True,
                    fill='none',
                    hovertemplate="%{text}<br>Quality: %{customdata:.0%}<br>Category: " + category + "<extra></extra>",
                    customdata=[1 - v for v in cat_normalized],
                    text=cat_hover
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
                period=len(formatted_param_names) if formatted_param_names else 1,
                rotation=90
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0,
            xanchor="left",
            x=1.2,
            title=dict(text="Categories")
        ),
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
    
    return fig, None