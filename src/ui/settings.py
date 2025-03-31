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
        
        # Scan for available cameras
        for i in range(5):  # Check first 5 camera indexes
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        cameras[i] = f"Camera {i}"
                    cap.release()
            except Exception:
                pass
                
        # If no cameras were found, use default from config
        if not cameras:
            for i, camera_id in enumerate(config.camera.camera_ids):
                cameras[camera_id] = f"Camera {camera_id}"
                
        # If still no cameras, add a default camera
        if not cameras:
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
                    # Set a default sampling rate that works with this device
                    # to avoid sample rate errors
                    default_samplerate = int(device.get('default_samplerate', 16000))
                    microphones[i] = {
                        'name': device['name'],
                        'sample_rate': default_samplerate
                    }
        except Exception as e:
            logger.error(f"Error getting audio devices: {str(e)}")
            
        # If no microphones were found, add a default one
        if not microphones:
            if config.speech.audio_device_index is not None:
                microphones[config.speech.audio_device_index] = {
                    'name': f"Microphone {config.speech.audio_device_index}",
                    'sample_rate': 16000
                }
            else:
                microphones[0] = {
                    'name': "Default Microphone",
                    'sample_rate': 16000
                }
                
        # Convert to format expected by existing code
        return {k: v['name'] for k, v in microphones.items()}

    
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