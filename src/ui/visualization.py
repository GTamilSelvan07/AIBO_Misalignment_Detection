"""
Visualization components for the Streamlit UI.
"""
import time
import math
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import streamlit as st
from typing import Dict, List, Optional, Tuple, Any
import altair as alt

from config import config


def plot_scores_history(histories: Dict[str, Tuple[List[float], List[int]]], 
                       title: str = "Misalignment Score History"):
    """
    Plot the history of misalignment scores.
    
    Args:
        histories: Dictionary of {person_name: (timestamps, scores)}
        title: Title for the plot
    """
    if not histories:
        return
        
    # Create a dataframe for altair
    df_data = []
    
    for person_name, (timestamps, scores) in histories.items():
        for ts, score in zip(timestamps, scores):
            df_data.append({
                "Timestamp": pd.to_datetime(ts, unit='s'),
                "Score": score,
                "Person": person_name
            })
            
    if not df_data:
        return
        
    df = pd.DataFrame(df_data)
    
    # Create line chart with altair
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X('Timestamp:T', title='Time'),
        y=alt.Y('Score:Q', title='Score', scale=alt.Scale(domain=[0, 100])),
        color=alt.Color('Person:N', title='Person'),
        tooltip=['Person', 'Score', 'Timestamp']
    ).properties(
        title=title,
        width=600,
        height=300
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)
    

def plot_scores_gauge(score: int, title: str = "Misalignment Score", 
                    thresholds: List[int] = [0, 20, 50, 80, 100]):
    """
    Plot a gauge chart for misalignment scores.
    
    Args:
        score: Misalignment score (0-100)
        title: Title for the gauge
        thresholds: Thresholds for color zones
    """
    # Create a figure
    fig, ax = plt.subplots(figsize=(3, 2), subplot_kw={'projection': 'polar'})
    
    # Convert score to radians (0-100 to 0-π)
    angle = np.pi * (score / 100)
    
    # Set up the gauge
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    
    # Set limits
    ax.set_rlim(0, 1)
    
    # Remove radial ticks and labels
    ax.set_rticks([])
    
    # Set up custom theta ticks and labels
    ax.set_xticks(np.linspace(0, np.pi, len(thresholds)))
    ax.set_xticklabels([str(t) for t in thresholds])
    
    # Create color bands
    cmap = plt.cm.RdYlGn_r
    norm = plt.Normalize(0, 100)
    
    # Draw colored arcs for different zones
    for i in range(len(thresholds) - 1):
        start_angle = np.pi * (thresholds[i] / 100)
        end_angle = np.pi * (thresholds[i+1] / 100)
        mid_angle = (start_angle + end_angle) / 2
        mid_score = (thresholds[i] + thresholds[i+1]) / 2
        color = cmap(norm(mid_score))
        
        ax.bar(
            mid_angle, 0.9, 
            width=end_angle-start_angle, 
            bottom=0.1, 
            color=color, 
            alpha=0.7
        )
    
    # Add needle
    ax.plot([0, angle], [0, 0.8], color='black', linewidth=2)
    
    # Add center circle
    circle = plt.Circle((0, 0), 0.1, transform=ax.transData._b, color='black', zorder=10)
    ax.add_artist(circle)
    
    # Remove border and background
    ax.spines['polar'].set_visible(False)
    ax.set_facecolor('none')
    
    # Add title
    plt.title(title, y=1.15, fontsize=10)
    
    # Add score text in the center
    plt.text(0, -0.2, f"{score}", ha='center', va='center', fontsize=12, fontweight='bold')
    
    st.pyplot(fig)
    

def highlight_misalignment_in_text(text: str, score: int) -> str:
    """
    Highlight text based on misalignment score.
    
    Args:
        text: Text to highlight
        score: Misalignment score (0-100)
        
    Returns:
        str: HTML-formatted text with appropriate highlighting
    """
    if score >= 80:
        return f'<div style="background-color:#ffcccc; padding:5px; border-radius:5px;">{text}</div>'
    elif score >= 50:
        return f'<div style="background-color:#ffe6cc; padding:5px; border-radius:5px;">{text}</div>'
    elif score >= 20:
        return f'<div style="background-color:#ffffcc; padding:5px; border-radius:5px;">{text}</div>'
    else:
        return text
    

def create_face_grid(frames: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Create a grid of face frames for display.
    
    Args:
        frames: Dictionary of {person_name: frame}
        
    Returns:
        np.ndarray: Combined grid image
    """
    if not frames:
        return np.zeros((100, 100, 3), dtype=np.uint8)
        
    # Extract frames, handling both (timestamp, frame) tuples and direct frames
    processed_frames = {}
    for name, frame_data in frames.items():
        if isinstance(frame_data, tuple) and len(frame_data) == 2:
            # This is a (timestamp, frame) tuple
            _, frame = frame_data
        else:
            # This is just a frame
            frame = frame_data
            
        if frame is not None:
            processed_frames[name] = frame
            
    if not processed_frames:
        return np.zeros((100, 100, 3), dtype=np.uint8)
        
    # Get frame dimensions
    sample_frame = next(iter(processed_frames.values()))
    height, width = sample_frame.shape[:2]
    
    # Resize frames if too large
    target_size = config.ui.camera_window_size
    if width > target_size[0] or height > target_size[1]:
        scale = min(target_size[0] / width, target_size[1] / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        for name in processed_frames:
            processed_frames[name] = cv2.resize(
                processed_frames[name], 
                (new_width, new_height)
            )
            
        width, height = new_width, new_height
        
    # Determine grid layout
    n_frames = len(processed_frames)
    
    if n_frames == 1:
        grid_cols = 1
        grid_rows = 1
    elif n_frames <= 2:
        grid_cols = 2
        grid_rows = 1
    elif n_frames <= 4:
        grid_cols = 2
        grid_rows = 2
    elif n_frames <= 6:
        grid_cols = 3
        grid_rows = 2
    else:
        grid_cols = math.ceil(math.sqrt(n_frames))
        grid_rows = math.ceil(n_frames / grid_cols)
        
    # Create the grid
    grid = np.zeros((grid_rows * height, grid_cols * width, 3), dtype=np.uint8)
    
    # Add frames to grid
    for i, (name, frame) in enumerate(processed_frames.items()):
        row = i // grid_cols
        col = i % grid_cols
        
        # Get frame region
        y_start = row * height
        y_end = (row + 1) * height
        x_start = col * width
        x_end = (col + 1) * width
        
        # Add frame to grid
        grid[y_start:y_end, x_start:x_end] = frame
        
        # Add name overlay
        cv2.putText(
            grid,
            name,
            (x_start + 10, y_start + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        
    return grid