"""
Streamlit UI module for the misalignment detection system.
"""
# Import components individually to avoid circular imports
from src.ui.visualizations import (
    plot_scores_history,
    plot_scores_gauge,
    highlight_misalignment_in_text,
    create_face_grid
)
from src.ui.settings import UISettings

# The app and components will be imported directly where needed
# to avoid circular import issues