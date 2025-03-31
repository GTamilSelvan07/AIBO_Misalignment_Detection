"""
Streamlit UI module for the misalignment detection system.
"""
from src.ui.app import run_app
from src.ui.components import (
    header_section, 
    camera_section, 
    speech_section, 
    combined_section,
    transcript_section,
    settings_section,
    status_section
)
from src.ui.visualizations import (
    plot_scores_history,
    plot_scores_gauge,
    highlight_misalignment_in_text,
    create_face_grid
)
from src.ui.settings import UISettings

__all__ = [
    'run_app',
    'header_section',
    'camera_section',
    'speech_section',
    'combined_section',
    'transcript_section',
    'settings_section',
    'status_section',
    'plot_scores_history',
    'plot_scores_gauge',
    'highlight_misalignment_in_text',
    'create_face_grid',
    'UISettings'
]