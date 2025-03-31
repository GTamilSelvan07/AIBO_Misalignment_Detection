"""
Configuration settings for the misalignment detection system.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Create directories if they don't exist
LOGS_DIR.mkdir(exist_ok=True)
(LOGS_DIR / "misalignment_scores").mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
(MODELS_DIR / "openface").mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Camera settings
class CameraConfig(BaseModel):
    camera_ids: List[int] = [0, 1]  # Default to first two cameras
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 30
    capture_interval: float = 5.0  # seconds
    detection_threshold: float = 0.7  # Face detection confidence threshold
    display_facial_landmarks: bool = True
    
    # Action Units (AUs) associated with confusion/misalignment
    confusion_aus: List[int] = [
        4,   # Brow lowerer
        7,   # Lid tightener
        9,   # Nose wrinkler
        14,  # Dimpler
        15,  # Lip corner depressor
        17,  # Chin raiser
        23,  # Lip tightener
        24   # Lip pressor
    ]
    
    # Weights for different AUs in confusion scoring
    au_weights: Dict[int, float] = {
        4: 1.5,   # Brow lowerer (strongest indicator)
        7: 1.2,   # Lid tightener
        9: 1.0,   # Nose wrinkler
        14: 0.8,  # Dimpler
        15: 1.0,  # Lip corner depressor
        17: 0.7,  # Chin raiser
        23: 0.9,  # Lip tightener
        24: 0.7   # Lip pressor
    }

# Speech settings
class SpeechConfig(BaseModel):
    audio_device_index: Optional[int] = None  # None means default device
    sample_rate: int = 16000
    channels: int = 1
    recording_interval: float = 5.0  # seconds
    energy_threshold: int = 300  # Minimum audio energy to consider as speech
    noise_duration: float = 1.0  # Duration for noise adjustment (seconds)
    phrase_threshold: float = 0.3  # Seconds of non-speaking audio to consider the phrase complete
    whisper_model: str = "base"  # Whisper model size (tiny, base, small, medium, large)

# LLM settings
class LLMConfig(BaseModel):
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model_name: str = os.getenv("OLLAMA_MODEL", "gemma3:1b")
    timeout: float = 10.0  # seconds
    max_tokens: int = 1024
    temperature: float = 0.1  # Lower temperature for more deterministic outputs
    top_p: float = 0.9
    top_k: int = 40
    
    # Misalignment detection thresholds
    confidence_threshold: float = 0.7
    min_score_for_alert: int = 50  # 0-100 range

# WebSocket settings
class WebSocketConfig(BaseModel):
    server_url: str = os.getenv("WEBSOCKET_SERVER", "ws://localhost:8765")
    reconnect_interval: float = 5.0  # seconds
    max_retries: int = 3
    ping_interval: float = 30.0  # seconds

# Scoring settings
class ScoringConfig(BaseModel):
    camera_weight: float = 0.5  # Weight for camera-based score in combined score
    speech_weight: float = 0.5  # Weight for speech-based score in combined score
    smoothing_window: int = 3    # Number of observations to use for smoothing
    alert_threshold: int = 70    # Combined score threshold for alerts (0-100)
    
    # Personalization parameters
    baseline_period: int = 60  # seconds to establish baseline behavior
    adaptation_rate: float = 0.05  # Rate at which to adapt to individual
    
    # History settings
    history_size: int = 100  # Number of observations to keep in history

# UI settings
class UIConfig(BaseModel):
    update_interval: float = 0.5  # seconds
    chart_history_size: int = 60  # Number of points to show in charts
    camera_window_size: tuple = (320, 240)  # (width, height)
    theme: str = "dark"
    highlight_color: str = "#FF6B6B"  # Color for highlighting misalignments
    
    # UI Layout sections
    show_raw_scores: bool = True
    show_combined_score: bool = True
    show_transcript: bool = True
    show_facial_features: bool = True
    show_settings: bool = True
    show_history: bool = True

# Logging settings
class LoggingConfig(BaseModel):
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: Path = LOGS_DIR / "app.log"
    console_logging: bool = True
    file_logging: bool = True
    log_rotation: str = "1 day"
    score_log_interval: float = 1.0  # seconds

# Combine all configurations
class AppConfig(BaseModel):
    camera: CameraConfig = CameraConfig()
    speech: SpeechConfig = SpeechConfig()
    llm: LLMConfig = LLMConfig()
    websocket: WebSocketConfig = WebSocketConfig()
    scoring: ScoringConfig = ScoringConfig()
    ui: UIConfig = UIConfig()
    logging: LoggingConfig = LoggingConfig()

# Create a singleton configuration instance
config = AppConfig()