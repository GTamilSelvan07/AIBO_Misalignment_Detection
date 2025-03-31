"""
Camera module for capturing and analyzing facial expressions.
"""
import os
import sys
from pathlib import Path

# Get the absolute path to the project root directory
current_dir = Path(__file__).resolve().parent.parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
    print("Project root added to sys.path:", current_dir)

# Now we can import our local modules with absolute imports
from src.camera.capture import CameraManager
from src.camera.face_detector import FaceDetector
from src.camera.misalignment import MisalignmentDetector

__all__ = ['CameraManager', 'FaceDetector', 'MisalignmentDetector']