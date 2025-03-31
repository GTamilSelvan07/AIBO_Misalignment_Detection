"""
Main Streamlit application for the misalignment detection system.
"""
import time
import threading
import streamlit as st
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger

from config import config
from src.camera import CameraManager, FaceDetector, MisalignmentDetector
from src.speech import AudioRecorder, SpeechTranscriber, SpeechAnalyzer
from src.llm import OllamaClient
from src.scoring import CameraScoreProcessor, SpeechScoreProcessor, CombinedScoreProcessor
from src.data import MisalignmentLogger, JsonGenerator, WebSocketClient
from src.ui.settings import UISettings

# Import components directly (don't import from src.ui to avoid circular imports)
from src.ui.components import (
    header_section,
    camera_section,
    speech_section,
    combined_section,
    transcript_section,
    settings_section,
    status_section
)


def initialize_system():
    """
    Initialize all system components.
    
    Returns:
        dict: Dictionary of component instances
    """
    logger.info("Initializing system components...")
    
    # Create component instances
    components = {
        "camera_manager": CameraManager(),
        "face_detector": FaceDetector(),
        "misalignment_detector": MisalignmentDetector(),
        "audio_recorder": AudioRecorder(),
        "speech_transcriber": SpeechTranscriber(),
        "llm_client": OllamaClient(),
        "speech_analyzer": None,  # Will be created after LLM client
        "camera_processor": CameraScoreProcessor(),
        "speech_processor": SpeechScoreProcessor(),
        "combined_processor": None,  # Will be created after camera and speech processors
        "misalignment_logger": MisalignmentLogger(),
        "json_generator": JsonGenerator(),
        "websocket_client": WebSocketClient(),
        "ui_settings": UISettings()
    }
    
    # Create speech analyzer with LLM client
    components["speech_analyzer"] = SpeechAnalyzer(components["llm_client"])
    
    # Create combined processor with camera and speech processors
    components["combined_processor"] = CombinedScoreProcessor(
        components["camera_processor"],
        components["speech_processor"]
    )
    
    # Set LLM client in combined processor
    components["combined_processor"].set_llm_client(components["llm_client"])
    
    # Wait for face detector initialization
    components["face_detector"].wait_for_initialization()
    
    logger.info("System components initialized")
    return components
    

def run_system(components: Dict, stop_event: threading.Event):
    """
    Run the main system loop.
    
    Args:
        components: Dictionary of component instances
        stop_event: Event to signal when to stop the system
    """
    logger.info("Starting system loop...")
    
    # Extract components
    camera_manager = components["camera_manager"]
    face_detector = components["face_detector"]
    misalignment_detector = components["misalignment_detector"]
    audio_recorder = components["audio_recorder"]
    speech_transcriber = components["speech_transcriber"]
    speech_analyzer = components["speech_analyzer"]
    camera_processor = components["camera_processor"]
    speech_processor = components["speech_processor"]
    combined_processor = components["combined_processor"]
    misalignment_logger = components["misalignment_logger"]
    json_generator = components["json_generator"]
    websocket_client = components["websocket_client"]
    ui_settings = components["ui_settings"]
    
    # Start cameras and audio
    camera_manager.start_all()
    audio_recorder.start_all_devices()
    
    # Start continuous capturing
    camera_manager.start_timed_capture(interval=ui_settings.capture_interval)
    audio_recorder.start_continuous_recording(interval=ui_settings.capture_interval)
    
    # Start transcription processing
    speech_transcriber.start_processing()
    
    # Start speech analysis processing
    speech_analyzer.start_processing()
    
    # Start logging
    misalignment_logger.start_logging()
    
    # Connect to WebSocket server if enabled
    if ui_settings.enable_websocket:
        websocket_client.connect()
    
    last_analysis_time = 0
    analysis_interval = ui_settings.capture_interval
    
    while not stop_event.is_set():
        current_time = time.time()
        
        # Check if it's time to analyze
        if current_time - last_analysis_time < analysis_interval:
            time.sleep(0.1)  # Short sleep to prevent tight loop
            continue
            
        # Update last analysis time
        last_analysis_time = current_time
        
        try:
            # Get camera frames
            timestamp, frames = camera_manager.get_next_synchronized_frames(timeout=0.5)
            
            if timestamp is None or not frames:
                continue
                
            # Detect faces in frames
            face_results = {}
            for person_name, (_, frame) in frames.items():
                if frame is not None:
                    success, face_data = face_detector.detect_face(frame)
                    face_results[person_name] = (success, face_data)
                    
            # Analyze faces for misalignment
            misalignment_results = misalignment_detector.analyze_multiple_faces(face_results)
            
            # Process camera scores
            for person_name, (score, details) in misalignment_results.items():
                camera_processor.add_score(person_name, score, details, timestamp)
                
            # Get audio chunks
            audio_results = {}
            for person_name in frames.keys():
                audio, energy, audio_timestamp = audio_recorder.get_audio_chunk(
                    person_name, 
                    duration=ui_settings.capture_interval
                )
                
                if audio is not None:
                    audio_results[person_name] = (audio, energy, audio_timestamp)
                    
            # Transcribe audio
            transcript_results = {}
            for person_name, (audio, _, audio_timestamp) in audio_results.items():
                # Queue for transcription
                item_id = speech_transcriber.queue_audio_for_transcription(
                    audio,
                    metadata={"person_name": person_name, "timestamp": audio_timestamp}
                )
                
                # Wait for result with timeout
                text, confidence, metadata = speech_transcriber.get_transcription_result(
                    item_id, 
                    timeout=2.0  # Wait up to 2 seconds for transcription
                )
                
                if text:
                    transcript_results[person_name] = (text, confidence, audio_timestamp)
                    
            # Analyze transcripts for misalignment
            for person_name, (text, confidence, text_timestamp) in transcript_results.items():
                # Add to conversation history and queue for analysis
                utterance_id, analysis_id = speech_analyzer.process_new_utterance(
                    person_name, text, text_timestamp, confidence
                )
                
                # Wait for analysis result with timeout
                score, details = speech_analyzer.get_analysis_result(
                    analysis_id,
                    timeout=2.0  # Wait up to 2 seconds for analysis
                )
                
                if score is not None:
                    # Process speech score
                    speech_processor.add_score(person_name, score, details, text, text_timestamp)
                    
            # Calculate combined scores
            for person_name in set(list(face_results.keys()) + list(transcript_results.keys())):
                # Get latest camera score
                camera_score, camera_details, _ = camera_processor.get_latest_score(person_name)
                if camera_score is None:
                    camera_score = 0
                    camera_details = {}
                    
                # Get latest speech score
                speech_score, speech_details, speech_text, _ = speech_processor.get_latest_score(person_name)
                if speech_score is None:
                    speech_score = 0
                    speech_details = {}
                    speech_text = ""
                    
                # Calculate combined score
                combined_score, combined_details = combined_processor.add_scores(
                    person_name,
                    camera_score, camera_details,
                    speech_score, speech_details, speech_text,
                    timestamp=current_time,
                    do_llm_analysis=ui_settings.enable_llm_analysis
                )
                
                # Log scores
                misalignment_logger.log_scores(
                    person_name, current_time,
                    combined_score, camera_score, speech_score,
                    combined_details
                )
                
                # Generate JSON and send via WebSocket if enabled
                if ui_settings.enable_websocket and websocket_client.is_connected:
                    json_data = json_generator.generate_misalignment_json(
                        person_name,
                        camera_score, camera_details,
                        speech_score, speech_details, speech_text,
                        combined_score,
                        timestamp=current_time
                    )
                    
                    websocket_client.send_misalignment_data(json_data)
                
        except Exception as e:
            logger.error(f"Error in system loop: {str(e)}")
            
    # Cleanup on exit
    logger.info("Stopping system loop...")
    
    # Stop components
    camera_manager.stop_all()
    audio_recorder.stop_all_devices()
    speech_transcriber.stop_processing()
    speech_analyzer.stop_processing()
    misalignment_logger.stop_logging()
    websocket_client.close()
    
    logger.info("System stopped")
    

def run_app():
    """
    Run the Streamlit application.
    """
    st.set_page_config(
        page_title="Misalignment Detection System",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state if needed
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        st.session_state.components = None
        st.session_state.system_running = False
        st.session_state.stop_event = threading.Event()
        st.session_state.system_thread = None
    
    # Display header
    header_section()
    
    # Initialize components if needed
    if not st.session_state.initialized:
        with st.spinner("Initializing system components..."):
            st.session_state.components = initialize_system()
            st.session_state.initialized = True
    
    # Create UI layout
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # Camera section
        camera_section(st.session_state.components)
        
        # Speech section
        speech_section(st.session_state.components)
    
    with col2:
        # Combined section
        combined_section(st.session_state.components)
        
        # Transcript section
        transcript_section(st.session_state.components)
    
    # Status and settings in sidebar
    with st.sidebar:
        # System control buttons
        st.subheader("System Control")
        
        cols = st.columns(2)
        with cols[0]:
            if not st.session_state.system_running:
                if st.button("Start System"):
                    st.session_state.system_running = True
                    st.session_state.stop_event.clear()
                    st.session_state.system_thread = threading.Thread(
                        target=run_system,
                        args=(st.session_state.components, st.session_state.stop_event),
                        daemon=True
                    )
                    st.session_state.system_thread.start()
                    st.success("System started")
            
        with cols[1]:
            if st.session_state.system_running:
                if st.button("Stop System"):
                    st.session_state.stop_event.set()
                    st.session_state.system_running = False
                    if st.session_state.system_thread:
                        st.session_state.system_thread.join(timeout=5.0)
                    st.warning("System stopped")
        
        # Settings section
        settings_section(st.session_state.components)
        
        # System status
        status_section(st.session_state.components)
    
    # Auto-refresh the app
    if st.session_state.system_running:
        time.sleep(config.ui.update_interval)
        st.experimental_rerun()


if __name__ == "__main__":
    run_app()