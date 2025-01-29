import streamlit as st

def create_parameter_tile(param_name, param_value, status='neutral'):
    """
    Create a parameter tile with colored value and status indicator.
    
    Args:
        param_name (str): Name of the parameter
        param_value (str): Value to display
        status (str): One of 'positive', 'negative', 'neutral', or 'untested'
    """
    # Define status colors and icons
    status_config = {
        'positive': ('✅', '#28a745'),  # Green
        'negative': ('⚠️', '#dc3545'),  # Red
        'neutral': ('', 'white'),      # White, no icon
        'untested': ('', '#6c757d')    # Grey, no icon
    }
    
    icon, color = status_config.get(status, status_config['untested'])
    
    # Create tile HTML with minimal styling
    tile_html = f"""
    <div style="margin: 10px 0; padding: 15px; border: 1px solid #dee2e6; border-radius: 5px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 0.9em; color: white;">{param_name}</div>
                <div style="font-size: 1.2em; color: {color}; margin-top: 5px;">
                    {icon} {param_value}
                </div>
            </div>
        </div>
    </div>
    """
    
    st.markdown(tile_html, unsafe_allow_html=True)

def create_parameter_tiles_grid(parameters, values, statuses=None, cols=3):
    """
    Create a grid of parameter tiles.
    
    Args:
        parameters (list): List of parameter names
        values (list): List of parameter values
        statuses (list): List of statuses ('positive', 'negative', 'neutral', 'untested')
        cols (int): Number of columns in the grid
    """
    if statuses is None:
        statuses = ['untested'] * len(parameters)
    
    # Create columns
    columns = st.columns(cols)
    
    # Distribute tiles across columns
    for idx, (param, value, status) in enumerate(zip(parameters, values, statuses)):
        # If the value is "Not Tested", override the status to 'untested'
        if value == "Not Tested":
            status = 'untested'
            
        with columns[idx % cols]:
            create_parameter_tile(param, value, status)

def create_collapsible_section(title, content_func):
    """
    Create a collapsible section with custom content.
    
    Args:
        title (str): Title of the collapsible section
        content_func (callable): Function to call to generate the content
    """
    with st.expander(title, expanded=False):
        content_func()