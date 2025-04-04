"""
Configuration settings for the misalignment detection system.
"""
import os
import json
from pathlib import Path

class Config:
    # Paths to external tools and models
    OPENFACE_PATH = r"C:\Users\tg469\Projects\PhD_Projects\AIBO\Dev\Misalignment_Detection\backend\models\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64"
    WHISPER_PATH = r"C:\Users\tg469\Projects\PhD_Projects\AIBO\Dev\Misalignment_Detection\backend\models\Faster-Whisper-XXL\faster-whisper-xxl.exe"
    
    # OpenFace executable and options
    OPENFACE_EXE = os.path.join(OPENFACE_PATH, "FeatureExtraction.exe")
    OPENFACE_OPTIONS = "-aus -gaze -2Dfp -3Dfp -pose -vis-track -vis-aus"
    
    # Data directories
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = os.path.join(BASE_DIR, "data")
    SESSIONS_DIR = os.path.join(DATA_DIR, "sessions")
    LOGS_DIR = os.path.join(DATA_DIR, "logs")
    
    # Ensure all directories exist
    for dir_path in [DATA_DIR, SESSIONS_DIR, LOGS_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Camera settings
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 10  # Frames per second
    CAMERA_DEVICE_IDS = [0, 1]  # Default camera device IDs
    
    # Audio settings
    AUDIO_SAMPLE_RATE = 48000
    AUDIO_CHANNELS = 1
    
    # Analysis settings
    ANALYSIS_INTERVAL_MS = 10000  # How often to run analysis (milliseconds)
    
    # LLM settings
    OLLAMA_URL = "http://localhost:11434/api/generate"
    GEMMA_MODEL = "gemma3:1b"
    
    # Websocket settings
    WEBSOCKET_HOST = "localhost"
    WEBSOCKET_PORT = 8765
    
    # UI settings
    UI_REFRESH_RATE_MS = 100
    UI_CHART_HISTORY = 100  # Number of data points to show in charts
    
    @classmethod
    def load_from_file(cls, file_path):
        """Load config from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                config_data = json.load(f)
                
            for key, value in config_data.items():
                if hasattr(cls, key):
                    setattr(cls, key, value)
                    
            return True
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            return False
    
    @classmethod
    def save_to_file(cls, file_path):
        """Save current config to a JSON file."""
        config_data = {}
        for key in dir(cls):
            if not key.startswith('__') and not callable(getattr(cls, key)):
                value = getattr(cls, key)
                # Skip Path objects or complex types that aren't JSON serializable
                if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                    config_data[key] = value
        
        try:
            with open(file_path, 'w') as f:
                json.dump(config_data, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving config: {str(e)}")
            return False