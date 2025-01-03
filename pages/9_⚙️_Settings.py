import streamlit as st
import json
from pathlib import Path

# Page config
st.set_page_config(page_title="Dashboard Settings", page_icon="⚙️", layout="wide")

def app():
    st.title("⚙️ Dashboard Settings")
    
    # Initialize settings from file or defaults
    settings_file = Path(__file__).parent.parent / "config" / "settings.json"
    settings_file.parent.mkdir(exist_ok=True)
    
    if settings_file.exists():
        settings = json.loads(settings_file.read_text())
    else:
        settings = {
            "zoom_levels": {
                "level1": 1.5,
                "level2": 2.0,
                "level3": 4.0
            }
        }
    
    st.header("Zoom Settings")
    st.markdown("""
    Configure the zoom levels for the radar charts. These settings will affect how the zoom buttons work in both the influent and treated water pages.
    
    For centered parameters like pH:
    - A zoom level of 1 shows the full range (e.g., 6.5-8.5)
    - A zoom level of 4 will show a narrower range around the middle (e.g., 7.25-7.75)
    
    For standard parameters like turbidity:
    - A zoom level of 1 shows the full range (0 to max)
    - A zoom level of 4 will show 0 to max/4
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        zoom1 = st.number_input(
            "Zoom Level 1",
            min_value=1.0,
            max_value=10.0,
            value=float(settings["zoom_levels"]["level1"]),
            step=0.1,
            help="First zoom level (default: 1.5x)"
        )
    
    with col2:
        zoom2 = st.number_input(
            "Zoom Level 2",
            min_value=1.0,
            max_value=10.0,
            value=float(settings["zoom_levels"]["level2"]),
            step=0.1,
            help="Second zoom level (default: 2x)"
        )
    
    with col3:
        zoom3 = st.number_input(
            "Zoom Level 3",
            min_value=1.0,
            max_value=10.0,
            value=float(settings["zoom_levels"]["level3"]),
            step=0.1,
            help="Third zoom level (default: 4x)"
        )
    
    # Save settings
    if st.button("Save Settings"):
        settings["zoom_levels"] = {
            "level1": zoom1,
            "level2": zoom2,
            "level3": zoom3
        }
        settings_file.write_text(json.dumps(settings, indent=2))
        st.success("Settings saved successfully!")
        
    # Preview
    st.header("Preview")
    st.markdown("### For pH (centered parameter, range 6.5-8.5):")
    for level, zoom in [("1", zoom1), ("2", zoom2), ("3", zoom3)]:
        mid = 7.5
        half_range = 1 / (2 * zoom)
        st.write(f"Level {level} (×{zoom:.1f}): {mid-half_range:.2f} - {mid+half_range:.2f}")
    
    st.markdown("### For Turbidity (standard parameter, max=10):")
    for level, zoom in [("1", zoom1), ("2", zoom2), ("3", zoom3)]:
        max_val = 10 / zoom
        st.write(f"Level {level} (×{zoom:.1f}): 0 - {max_val:.2f}")

# For testing locally
if __name__ == "__main__":
    app()