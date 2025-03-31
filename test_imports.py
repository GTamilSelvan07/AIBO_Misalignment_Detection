#!/usr/bin/env python3
"""
Test script to verify that all imports are working correctly.
"""
import os
import sys
from pathlib import Path

# Add the project root directory to the path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
    print("Project root added to sys.path:", current_dir)

# Make sure the src directory is also in the path
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
    print("src directory added to sys.path:", src_dir)

def test_imports():
    """
    Test all the main imports to ensure they're working correctly.
    """
    print("Testing imports...")
    
    # Import test blocks wrapped in try/except to identify specific issues
    try:
        from config import config
        print("✅ Successfully imported config")
    except ImportError as e:
        print(f"❌ Failed to import config: {e}")
    
    try:
        from src.camera import CameraManager, FaceDetector, MisalignmentDetector
        print("✅ Successfully imported camera modules")
    except ImportError as e:
        print(f"❌ Failed to import camera modules: {e}")
    
    try:
        from src.speech import AudioRecorder, SpeechTranscriber, SpeechAnalyzer
        print("✅ Successfully imported speech modules")
    except ImportError as e:
        print(f"❌ Failed to import speech modules: {e}")
    
    try:
        from src.llm import OllamaClient
        print("✅ Successfully imported LLM modules")
    except ImportError as e:
        print(f"❌ Failed to import LLM modules: {e}")
    
    try:
        from src.scoring import CameraScoreProcessor, SpeechScoreProcessor, CombinedScoreProcessor
        print("✅ Successfully imported scoring modules")
    except ImportError as e:
        print(f"❌ Failed to import scoring modules: {e}")
    
    try:
        from src.data import setup_logging, MisalignmentLogger, JsonGenerator, WebSocketClient
        print("✅ Successfully imported data modules")
    except ImportError as e:
        print(f"❌ Failed to import data modules: {e}")
    
    try:
        from src.ui import run_app
        print("✅ Successfully imported UI modules")
    except ImportError as e:
        print(f"❌ Failed to import UI modules: {e}")
    
    print("\nVerifying file existence...")
    files_to_check = [
        "config.py",
        "src/camera/capture.py",
        "src/camera/face_detector.py",
        "src/camera/misalignment.py",
        "src/speech/recorder.py",
        "src/speech/transcriber.py",
        "src/speech/analysis.py",
        "src/llm/ollama_client.py",
        "src/llm/prompts.py",
        "src/llm/response_parser.py",
        "src/scoring/camera_score.py",
        "src/scoring/speech_score.py",
        "src/scoring/combined_score.py",
        "src/data/logger.py",
        "src/data/json_generator.py",
        "src/data/websocket.py",
        "src/ui/app.py"
    ]
    
    for file_path in files_to_check:
        full_path = current_dir / file_path
        if full_path.exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} does not exist")

if __name__ == "__main__":
    test_imports()
    print("\nImport test completed.")