"""
Error handling and logging utilities for the application.
"""
import logging
import os
import sys
import traceback
from datetime import datetime
from .config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(Config.LOGS_DIR, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    ]
)

def get_logger(name):
    """Get a logger configured for the application."""
    return logging.getLogger(name)

class ApplicationError(Exception):
    """Base class for application-specific exceptions."""
    def __init__(self, message, module=None, error_code=None):
        self.message = message
        self.module = module
        self.error_code = error_code
        self.timestamp = datetime.now()
        super().__init__(self.message)
    
    def __str__(self):
        return f"[{self.module}] {self.message} (Code: {self.error_code})"

class CameraError(ApplicationError):
    """Errors related to camera operations."""
    def __init__(self, message, error_code=None):
        super().__init__(message, module="CameraManager", error_code=error_code)

class AudioError(ApplicationError):
    """Errors related to audio operations."""
    def __init__(self, message, error_code=None):
        super().__init__(message, module="AudioManager", error_code=error_code)

class LLMError(ApplicationError):
    """Errors related to LLM operations."""
    def __init__(self, message, error_code=None):
        super().__init__(message, module="LLMAnalyzer", error_code=error_code)

class WebSocketError(ApplicationError):
    """Errors related to WebSocket operations."""
    def __init__(self, message, error_code=None):
        super().__init__(message, module="WebSocketServer", error_code=error_code)

def log_exception(logger, e, additional_info=None):
    """Log an exception with stack trace and additional info."""
    error_msg = f"Exception: {str(e)}"
    if additional_info:
        error_msg += f" | Additional Info: {additional_info}"
    
    logger.error(error_msg)
    logger.error(traceback.format_exc())
    
    return error_msg