"""
Settings management for the UI.
"""
from config import config


class UISettings:
    """
    Manages settings for the UI.
    """
    def __init__(self):
        """
        Initialize UI settings with defaults from config.
        """
        # Capture settings
        self.capture_interval = config.camera.capture_interval
        self.energy_threshold = config.speech.energy_threshold
        
        # Scoring settings
        self.camera_weight = config.scoring.camera_weight
        self.speech_weight = config.scoring.speech_weight
        self.alert_threshold = config.scoring.alert_threshold
        
        # LLM settings
        self.enable_llm_analysis = True
        
        # WebSocket settings
        self.enable_websocket = False
        self.websocket_url = config.websocket.server_url