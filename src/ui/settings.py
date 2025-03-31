"""
Settings management for the UI.
"""
import cv2
import sounddevice as sd
from typing import Dict, List, Tuple
from loguru import logger
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
        
        # Camera and microphone selection
        self.selected_cameras = self._get_default_cameras()
        self.selected_microphones = self._get_default_microphones()
        
        # CSV export settings
        self.csv_export_dir = config.LOGS_DIR / "exports"
        self.csv_export_interval = 60.0  # Export every minute
        self.enable_csv_export = True
        
    def _get_default_cameras(self) -> Dict[int, str]:
        """
        Get a dictionary of available cameras with their IDs.
        
        Returns:
            dict: Dictionary of {camera_id: camera_name}
        """
        cameras = {}
        
        # Try to get the default cameras from config
        for i, camera_id in enumerate(config.camera.camera_ids):
            cameras[camera_id] = f"Camera {camera_id}"
        
        if not cameras:
            # Fallback to default camera
            cameras[0] = "Default Camera"
            
        return cameras
        
    def _get_default_microphones(self) -> Dict[int, str]:
        """
        Get a dictionary of available microphones with their IDs.
        
        Returns:
            dict: Dictionary of {microphone_id: microphone_name}
        """
        microphones = {}
        
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    microphones[i] = device['name']
        except Exception as e:
            logger.error(f"Error getting audio devices: {str(e)}")
            # Fallback to default microphone
            if config.speech.audio_device_index is not None:
                microphones[config.speech.audio_device_index] = f"Microphone {config.speech.audio_device_index}"
            else:
                microphones[0] = "Default Microphone"
                
        if not microphones:
            # Ensure at least one microphone is available
            microphones[0] = "Default Microphone"
            
        return microphones
    
    def get_all_cameras(self) -> Dict[int, str]:
        """
        Get a dictionary of all available cameras.
        
        Returns:
            dict: Dictionary of {camera_id: camera_name}
        """
        cameras = {}
        
        # Try to detect available cameras (this is system dependent)
        num_cameras = 10  # Check first 10 camera indexes
        
        for i in range(num_cameras):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        cameras[i] = f"Camera {i}"
                    cap.release()
            except Exception:
                pass
                
        if not cameras:
            # Fallback to default camera
            cameras[0] = "Default Camera"
            
        return cameras
    
    def get_all_microphones(self) -> Dict[int, str]:
        """
        Get a dictionary of all available microphones.
        
        Returns:
            dict: Dictionary of {microphone_id: microphone_name}
        """
        microphones = {}
        
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    microphones[i] = device['name']
        except Exception as e:
            logger.error(f"Error getting audio devices: {str(e)}")
            # Fallback to default microphone
            microphones[0] = "Default Microphone"
            
        if not microphones:
            # Ensure at least one microphone is available
            microphones[0] = "Default Microphone"
            
        return microphones