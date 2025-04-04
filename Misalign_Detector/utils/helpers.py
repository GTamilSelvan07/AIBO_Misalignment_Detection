"""
Helper utilities for the misalignment detection system.
"""
import os
import uuid
import json
import time
import csv
from datetime import datetime
from .config import Config
from .error_handler import get_logger

logger = get_logger(__name__)

def generate_session_id():
    """Generate a unique session ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"session_{timestamp}_{unique_id}"

def create_session_directory(session_id):
    """Create a directory for storing session data."""
    session_dir = os.path.join(Config.SESSIONS_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    # Create subdirectories for different data types
    for subdir in ['video', 'audio', 'transcripts', 'analysis', 'features']:
        os.makedirs(os.path.join(session_dir, subdir), exist_ok=True)
    
    return session_dir

def save_json(data, file_path):
    """Save data as JSON to file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON: {str(e)}")
        return False

def load_json(file_path):
    """Load JSON data from file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON: {str(e)}")
        return None

def append_to_csv(data_row, file_path, headers=None):
    """Append a row of data to a CSV file."""
    try:
        file_exists = os.path.isfile(file_path)
        
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Write header if file doesn't exist and headers are provided
            if not file_exists and headers:
                writer.writerow(headers)
            
            writer.writerow(data_row)
        return True
    except Exception as e:
        logger.error(f"Error appending to CSV: {str(e)}")
        return False

def format_timestamp(timestamp=None):
    """Format a timestamp for display or logging."""
    if timestamp is None:
        timestamp = time.time()
    
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

def extract_emotions_from_aus(action_units):
    """
    Extract emotion scores from facial action units (AUs).
    
    This is a simplified mapping of AUs to emotions based on FACS.
    For a real system, you would want a more sophisticated model.
    
    Args:
        action_units (dict): Dictionary with AU codes as keys and intensities as values
        
    Returns:
        dict: Dictionary of emotion scores
    """
    # Simplified emotion mappings to Action Units
    emotion_mappings = {
        "confusion": ["AU4", "AU7", "AU23"],  # Brow lowerer, lid tightener, lip tightener
        "interest": ["AU1", "AU2", "AU5"],    # Inner brow raiser, outer brow raiser, upper lid raiser
        "frustration": ["AU4", "AU5", "AU7", "AU23"],  # Brow lowerer, upper lid raiser, lid tightener, lip tightener
        "understanding": ["AU1", "AU2", "AU6", "AU12"]  # Inner brow raiser, outer brow raiser, cheek raiser, lip corner puller
    }
    
    emotions = {}
    
    # Calculate each emotion score based on the average intensity of its associated AUs
    for emotion, aus in emotion_mappings.items():
        au_values = [action_units.get(au, 0) for au in aus]
        if au_values:
            emotions[emotion] = sum(au_values) / len(au_values)
        else:
            emotions[emotion] = 0
    
    return emotions

def normalize_score(score, min_val=0, max_val=1):
    """Normalize a score to be between min_val and max_val."""
    if score < min_val:
        return min_val
    if score > max_val:
        return max_val
    return score