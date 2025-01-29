import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
from utils.data_loader import load_all_data
from utils.functions import (
    calculate_change, process_sequence_states, 
    calculate_state_metrics, prepare_state_distribution_data,
    calculate_state_transitions, calculate_state_durations
)

# Page config
st.set_page_config(page_title="Overview Dashboard", page_icon="📊", layout="wide")

def calculate_change(current, previous):
    """Calculate percentage change between periods"""
    if previous == 0:
        return 0
    return ((current - previous) / previous) * 100

def generate_dummy_energy_data(start_date, days=30):
    """Generate dummy energy consumption data"""
    # Use pd.date_range for more efficient date generation
    hourly_dates = pd.date_range(start=start_date, periods=days*24, freq='H')
    hourly_usage = []
    
    for dt in hourly_dates:
        hour = dt.hour
        base_load = random.uniform(80, 120)
        time_factor = 1.0
        if 9 <= hour <= 17:  # Higher usage during working hours
            time_factor = 1.5
        elif 0 <= hour <= 5:  # Lower usage during night
            time_factor = 0.6
        hourly_usage.append(base_load * time_factor)
    
    return pd.DataFrame({
        'datetime': hourly_dates,
        'kw_usage': hourly_usage,
        'date': hourly_dates.date
    })

def generate_dummy_production_data(start_date, days=30):
    """Generate dummy water production data"""
    dates = [start_date + timedelta(days=x) for x in range(days)]
    return pd.DataFrame({
        'date': dates,
        'water_treated': [random.uniform(8000, 12000) for _ in range(days)],
        'water_consumed': [random.uniform(200, 400) for _ in range(days)],
        'water_quality': [random.uniform(90, 99) for _ in range(days)],
        'pressure': [random.uniform(45, 55) for _ in range(days)]
    })

def create_energy_metrics(energy_df):
    """Create energy consumption metrics and charts"""
    # Calculate period metrics
    current_daily_avg = energy_df.loc[energy_df['datetime'] >= (energy_df['datetime'].max() - timedelta(days=7)), 'kw_usage'].mean()
    previous_daily_avg = energy_df.loc[(energy_df['datetime'] >= (energy_df['datetime'].max() - timedelta(days=14))) & 
                                     (energy_df['datetime'] < (energy_df['datetime'].max() - timedelta(days=7))), 'kw_usage'].mean()
    change = calculate_change(current_daily_avg, previous_daily_avg)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Daily Average Usage", 
                 f"{current_daily_avg:.1f} kW",
                 f"{change:+.1f}% vs previous week",
                 delta_color="inverse")
    
    peak_usage = energy_df['kw_usage'].max()
    previous_peak = energy_df.loc[energy_df['datetime'] < (energy_df['datetime'].max() - timedelta(days=7)), 'kw_usage'].max()
    peak_change = calculate_change(peak_usage, previous_peak)
    
    with col2:
        st.metric("Peak Usage",
                 f"{peak_usage:.1f} kW",
                 f"{peak_change:+.1f}% vs previous week",
                 delta_color="inverse")
    
    with col3:
        # Fix: Combine ranges properly using list
        off_peak_hours = list(range(22, 24)) + list(range(0, 6))
        off_peak_avg = energy_df.loc[energy_df['datetime'].dt.hour.isin(off_peak_hours), 'kw_usage'].mean()
        st.metric("Off-Peak Average",
                 f"{off_peak_avg:.1f} kW")

    # Daily usage column chart
    daily_usage = energy_df.groupby('date')['kw_usage'].mean().reset_index()
    fig_daily = go.Figure(data=[
        go.Bar(x=daily_usage['date'],
               y=daily_usage['kw_usage'],
               name='Daily Usage',
               marker_color='#2E86C1')
    ])
    fig_daily.update_layout(
        title='Daily Energy Usage',
        xaxis_title='Date',
        yaxis_title='Average kW',
        height=400
    )
    st.plotly_chart(fig_daily, use_container_width=True)

    # Hourly usage pattern
    col1, col2 = st.columns(2)
    
    with col1:
        # 24-hour usage pattern
        hourly_avg = energy_df.groupby(energy_df['datetime'].dt.hour)['kw_usage'].mean()
        fig_hourly = go.Figure(data=[
            go.Scatter(x=hourly_avg.index,
                      y=hourly_avg.values,
                      fill='tozeroy',
                      name='Average Usage',
                      line=dict(color='#2E86C1'))
        ])
        fig_hourly.update_layout(
            title='24-Hour Usage Pattern',
            xaxis_title='Hour of Day',
            yaxis_title='Average kW',
            height=300
        )
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    with col2:
        # Usage by day of week
        dow_avg = energy_df.groupby(energy_df['datetime'].dt.day_name())['kw_usage'].mean()
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_avg = dow_avg.reindex(days_order)
        
        fig_dow = go.Figure(data=[
            go.Bar(x=dow_avg.index,
                  y=dow_avg.values,
                  marker_color='#2E86C1')
        ])
        fig_dow.update_layout(
            title='Usage by Day of Week',
            xaxis_title='Day',
            yaxis_title='Average kW',
            height=300
        )
        st.plotly_chart(fig_dow, use_container_width=True)

def create_efficiency_metrics(telemetry_df):
    """Create and display system efficiency metrics and visualizations based on telemetry data"""
    
    def categorize_state(state_name):
        """Categorize states into main categories with subcategories"""
        state_lower = state_name.lower()
        
        # Production states
        if 'production' in state_lower:
            return 'Production', 'Main Production'
        elif 'uvlamp' in state_lower:
            return 'Production', 'UV Treatment'
        elif 'permeability' in state_lower:
            return 'Production', 'Permeability Test'
            
        # Cleaning states
        elif any(x in state_lower for x in ['flush', 'clean', 'scour', 'backwash']):
            if 'prefilter' in state_lower:
                return 'Cleaning', 'Prefilter Cleaning'
            elif 'system' in state_lower:
                return 'Cleaning', 'System Flush'
            else:
                return 'Cleaning', 'Membrane Cleaning'
                
        # Testing states
        elif 'test' in state_lower:
            if 'leakage' in state_lower:
                return 'Testing', 'Leakage Test'
            elif 'integrity' in state_lower:
                return 'Testing', 'Integrity Test'
            else:
                return 'Testing', 'Other Tests'
                
        # System states
        elif any(x in state_lower for x in ['initialization', 'wait', 'stop', 'alarm', 'health']):
            return 'System', state_name
            
        # Default/Unknown
        return 'Other', state_name

    def process_telemetry_sequences(df):
        """Process telemetry data into sequence intervals using packet counting"""
        df = df.sort_values('timestamp')
        
        # Group by sequence and count packets
        sequence_counts = df.groupby(['FLOWSEQUENCE', df['timestamp'].dt.date]).size().reset_index()
        sequence_counts.columns = ['state_name', 'date', 'packet_count']
        
        # Calculate duration (10 seconds per packet)
        sequence_counts['duration'] = sequence_counts['packet_count'] * 10 / 60  # Convert to minutes
        
        # Add categories
        sequence_counts[['category', 'sub_category']] = sequence_counts.apply(
            lambda x: pd.Series(categorize_state(x['state_name'])), axis=1
        )
        
        # Calculate average metrics per state
        sequences = []
        for _, row in sequence_counts.iterrows():
            state_data = df[df['FLOWSEQUENCE'] == row['state_name']]
            
            sequences.append({
                'state_name': row['state_name'],
                'date': row['date'],
                'category': row['category'],
                'sub_category': row['sub_category'],
                'duration': row['duration'],
                'packet_count': row['packet_count'],
                'avg_pressure': state_data['PTC101_PRESSURE'].mean(),
                'avg_flow': state_data['FTR102_FLOWRATE'].mean()
            })
        
        return pd.DataFrame(sequences)
        
        # Add the last sequence
        if current_sequence is not None:
            duration = (pd.to_datetime(df['timestamp'].iloc[-1]) - pd.to_datetime(start_time)).total_seconds() / 60
            main_category, sub_category = categorize_state(current_sequence)
            sequences.append({
                'state_name': current_sequence,
                'start_time': start_time,
                'end_time': df['timestamp'].iloc[-1],
                'category': main_category,
                'sub_category': sub_category,
                'duration': duration,
                'avg_pressure': df[df['timestamp'] >= start_time]['PTC101_PRESSURE'].mean(),
                'avg_flow': df[df['timestamp'] >= start_time]['FTR102_FLOWRATE'].mean()
            })
        
        return pd.DataFrame(sequences)

    # Process sequences
    sequences_df = process_telemetry_sequences(telemetry_df)
    sequences_df['start_time'] = pd.to_datetime(sequences_df['start_time'])
    sequences_df['end_time'] = pd.to_datetime(sequences_df['end_time'])
    
    # Calculate high-level metrics
    last_24h = sequences_df[sequences_df['end_time'] >= (sequences_df['end_time'].max() - pd.Timedelta(hours=24))]
    last_7d = sequences_df[sequences_df['end_time'] >= (sequences_df['end_time'].max() - pd.Timedelta(days=7))]
    
    def calculate_period_metrics(period_df):
        total_duration = period_df['duration'].sum()
        production_time = period_df[period_df['category'] == 'Production']['duration'].sum()
        cleaning_time = period_df[period_df['category'] == 'Cleaning']['duration'].sum()
        
        return {
            'production_pct': (production_time / total_duration * 100) if total_duration > 0 else 0,
            'cleaning_pct': (cleaning_time / total_duration * 100) if total_duration > 0 else 0,
            'avg_production_flow': period_df[period_df['category'] == 'Production']['avg_flow'].mean(),
            'state_changes': len(period_df)
        }
    
    current_metrics = calculate_period_metrics(last_24h)
    previous_metrics = calculate_period_metrics(last_7d)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Production Uptime",
            f"{current_metrics['production_pct']:.1f}%",
            f"{current_metrics['production_pct'] - previous_metrics['production_pct']:.1f}%"
        )
    
    with col2:
        st.metric(
            "Average Production Flow",
            f"{current_metrics['avg_production_flow']:.1f} L/min",
            f"{(current_metrics['avg_production_flow'] - previous_metrics['avg_production_flow']):.1f} L/min"
        )
    
    with col3:
        st.metric(
            "State Changes (24h)",
            f"{current_metrics['state_changes']}",
            f"{current_metrics['state_changes'] - previous_metrics['state_changes']}"
        )
    
    # Create state distribution charts
    colors = {
        'Production': '#2ECC71',
        'Cleaning': '#E74C3C',
        'Testing': '#F1C40F',
        'System': '#3498DB',
        'Other': '#95A5A6'
    }
    
    # Time-based distribution (in hours)
    daily_duration = sequences_df.groupby(['date', 'category'])['duration'].sum().reset_index()
    daily_duration_wide = daily_duration.pivot(index='date', columns='category', values='duration').fillna(0)
    
    fig_distribution = go.Figure()
    for category in daily_duration_wide.columns:
        fig_distribution.add_trace(go.Bar(
            name=category,
            x=daily_duration_wide.index,
            y=daily_duration_wide[category] / 60,  # Convert to hours
            marker_color=colors.get(category, '#95A5A6')
        ))
    
    fig_distribution.update_layout(
        barmode='stack',
        title='Daily State Distribution (Hours)',
        xaxis_title='Date',
        yaxis_title='Duration (hours)',
        height=300,
        hovermode='x unified'
    )
    st.plotly_chart(fig_distribution, use_container_width=True)
    
    # Normalized distribution (ratios)
    daily_totals = daily_duration_wide.sum(axis=1)
    normalized_duration = daily_duration_wide.div(daily_totals, axis=0) * 100
    
    fig_normalized = go.Figure()
    for category in normalized_duration.columns:
        fig_normalized.add_trace(go.Bar(
            name=category,
            x=normalized_duration.index,
            y=normalized_duration[category],
            marker_color=colors.get(category, '#95A5A6')
        ))
    
    fig_normalized.update_layout(
        barmode='stack',
        title='Daily State Distribution (Normalized)',
        xaxis_title='Date',
        yaxis_title='Percentage (%)',
        height=300,
        hovermode='x unified'
    )
    st.plotly_chart(fig_normalized, use_container_width=True)
    
    # Create detailed state analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Average duration by state category
        avg_duration = sequences_df.groupby('sub_category')['duration'].mean().reset_index()
        avg_duration = avg_duration.sort_values('duration', ascending=True)
        
        fig_duration = go.Figure(data=[
            go.Bar(
                y=avg_duration['sub_category'],
                x=avg_duration['duration'],
                orientation='h',
                marker_color='#3498DB'
            )
        ])
        
        fig_duration.update_layout(
            title='Average Duration by State',
            xaxis_title='Duration (minutes)',
            yaxis_title='State',
            height=400
        )
        st.plotly_chart(fig_duration, use_container_width=True)
    
    with col2:
        # State transition patterns
        state_transitions = pd.DataFrame({
            'from_state': sequences_df['sub_category'].iloc[:-1].values,
            'to_state': sequences_df['sub_category'].iloc[1:].values,
            'count': 1
        })
        
        transition_counts = state_transitions.groupby(['from_state', 'to_state'])['count'].sum().reset_index()
        transition_counts = transition_counts.sort_values('count', ascending=False).head(10)
        
        fig_transitions = go.Figure(data=[
            go.Bar(
                x=transition_counts['count'],
                y=[f"{row['from_state']} → {row['to_state']}" for _, row in transition_counts.iterrows()],
                orientation='h',
                marker_color='#E67E22'
            )
        ])
        
        fig_transitions.update_layout(
            title='Top 10 State Transitions',
            xaxis_title='Number of Transitions',
            yaxis_title='Transition Pattern',
            height=400
        )
        st.plotly_chart(fig_transitions, use_container_width=True)
    
    # Add timeline view
    st.subheader("Recent State Timeline")
    recent_states = sequences_df.tail(50)  # Show last 50 states
    
    fig_timeline = go.Figure()
    
    for category in colors.keys():
        category_data = recent_states[recent_states['category'] == category]
        if not category_data.empty:
            fig_timeline.add_trace(go.Scatter(
                x=category_data['start_time'],
                y=[category] * len(category_data),
                mode='markers',
                name=category,
                marker=dict(
                    color=colors[category],
                    size=10
                ),
                hovertext=[f"{row['sub_category']}<br>Duration: {row['duration']:.1f} min" 
                          for _, row in category_data.iterrows()],
                hoverinfo='text+x'
            ))
    
    fig_timeline.update_layout(
        title='State Timeline',
        xaxis_title='Time',
        yaxis_title='Category',
        height=300,
        showlegend=True,
        hovermode='closest'
    )
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Add efficiency insights
    st.subheader("System Efficiency Insights")
    
    # Calculate insights
    total_transitions = len(sequences_df)
    avg_state_duration = sequences_df['duration'].mean()
    most_common_state = sequences_df['sub_category'].mode().iloc[0]
    production_pressure = sequences_df[sequences_df['category'] == 'Production']['avg_pressure'].mean()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        📊 System Statistics:
        - Total state transitions: {total_transitions}
        - Average state duration: {avg_state_duration:.1f} minutes
        - Most common state: {most_common_state}
        - Average production pressure: {production_pressure:.1f} PSI
        """)
    
    with col2:
        # Calculate recent anomalies or patterns
        short_states = sequences_df[sequences_df['duration'] < 1].shape[0]
        cleaning_frequency = sequences_df[sequences_df['category'] == 'Cleaning'].shape[0]
        
        st.warning(f"""
        ⚠️ System Patterns:
        - Short duration states (<1 min): {short_states}
        - Cleaning cycles in period: {cleaning_frequency}
        - Total packets processed: {sequences_df['packet_count'].sum():,}
        """)

def create_production_metrics(production_df):
    """Create water production metrics and charts"""
    # Calculate period metrics
    current_treated = production_df['water_treated'].tail(7).mean()
    previous_treated = production_df['water_treated'].iloc[-14:-7].mean()
    treated_change = calculate_change(current_treated, previous_treated)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Daily Water Treated",
                 f"{current_treated:.0f} L",
                 f"{treated_change:+.1f}% vs previous week")
    
    current_consumed = production_df['water_consumed'].tail(7).mean()
    previous_consumed = production_df['water_consumed'].iloc[-14:-7].mean()
    consumed_change = calculate_change(current_consumed, previous_consumed)
    
    with col2:
        st.metric("Daily Water Consumed",
                 f"{current_consumed:.0f} L",
                 f"{consumed_change:+.1f}% vs previous week",
                 delta_color="inverse")
    
    current_quality = production_df['water_quality'].tail(7).mean()
    previous_quality = production_df['water_quality'].iloc[-14:-7].mean()
    quality_change = calculate_change(current_quality, previous_quality)
    
    with col3:
        st.metric("Water Quality",
                 f"{current_quality:.1f}%",
                 f"{quality_change:+.1f}% vs previous week")

    # Production vs Consumption Chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=production_df['date'],
        y=production_df['water_treated'],
        name='Water Treated',
        line=dict(color='#1E90FF', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=production_df['date'],
        y=production_df['water_consumed'],
        name='Water Consumed',
        line=dict(color='#8B4513', width=2)
    ))
    
    fig.update_layout(
        title='Daily Water Production vs Consumption',
        xaxis_title='Date',
        yaxis_title='Liters',
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Additional charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Water quality trend
        fig_quality = go.Figure()
        fig_quality.add_trace(go.Scatter(
            x=production_df['date'],
            y=production_df['water_quality'],
            fill='tozeroy',
            name='Quality',
            line=dict(color='#1E90FF')
        ))
        fig_quality.update_layout(
            title='Water Quality Trend',
            xaxis_title='Date',
            yaxis_title='Quality (%)',
            height=300,
            yaxis=dict(range=[85, 100])  # Adjust range to better show quality variations
        )
        st.plotly_chart(fig_quality, use_container_width=True)
    
    with col2:
        # System pressure trend
        fig_pressure = go.Figure()
        fig_pressure.add_trace(go.Scatter(
            x=production_df['date'],
            y=production_df['pressure'],
            name='Pressure',
            line=dict(color='#2ECC71')
        ))
        fig_pressure.add_trace(go.Scatter(
            x=production_df['date'],
            y=[50] * len(production_df),  # Target pressure line
            name='Target',
            line=dict(color='#E74C3C', dash='dash')
        ))
        fig_pressure.update_layout(
            title='System Pressure Monitoring',
            xaxis_title='Date',
            yaxis_title='Pressure (PSI)',
            height=300,
            showlegend=True
        )
        st.plotly_chart(fig_pressure, use_container_width=True)

    # Add efficiency ratio calculation
    production_df['efficiency_ratio'] = production_df['water_treated'] / (production_df['water_consumed'] + 1)  # Add 1 to avoid division by zero
    
    # Efficiency ratio trend
    fig_efficiency = go.Figure()
    fig_efficiency.add_trace(go.Scatter(
        x=production_df['date'],
        y=production_df['efficiency_ratio'],
        fill='tozeroy',
        name='Efficiency Ratio',
        line=dict(color='#9B59B6')
    ))
    fig_efficiency.update_layout(
        title='Production Efficiency Ratio Trend',
        xaxis_title='Date',
        yaxis_title='Efficiency Ratio',
        height=300
    )
    st.plotly_chart(fig_efficiency, use_container_width=True)

def main():
    """Main function to run the dashboard"""
    st.title("Water Treatment Plant Dashboard")
    st.markdown("---")
    
    # Load all data
    data = load_all_data()
    
    # Get sequence data from loaded data
    telemetry_df  = data['telemetry']
    
    ### Generate dummy data ###
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    energy_df = generate_dummy_energy_data(start_date)
    production_df = generate_dummy_production_data(start_date)
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Production Metrics", "Energy Usage", "System Efficiency"])
    
    with tab1:
        st.header("Water Production Overview")
        create_production_metrics(production_df)
    
    with tab2:
        st.header("Energy Consumption Analysis")
        create_energy_metrics(energy_df)
    
    with tab3:
        st.header("System Efficiency Analysis")
        create_efficiency_metrics(telemetry_df)
    
    # Add footer with last update time
    st.markdown("---")
    st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

if __name__ == "__main__":
    main()
