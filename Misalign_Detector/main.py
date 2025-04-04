#!/usr/bin/env python3
"""
Main entry point for the misalignment detection system.
"""
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import Config
from utils.error_handler import get_logger, log_exception
from ui.main_app import run_app

logger = get_logger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Misalignment Detection System")
    
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to config file"
    )
    
    parser.add_argument(
        "--log-level", 
        type=str, 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Log level"
    )
    
    parser.add_argument(
        "--headless", 
        action="store_true", 
        help="Run in headless mode (no UI, API only)"
    )
    
    return parser.parse_args()

def run_headless_mode():
    """Run the system in headless mode (no UI, API only)."""
    try:
        logger.info("Starting in headless mode...")
        
        # Import components for headless mode
        from utils.helpers import generate_session_id, create_session_directory
        from core.camera_manager import CameraManager
        from core.audio_manager import AudioManager
        from core.llm_analyzer import LLMAnalyzer
        from core.detector import MisalignmentDetector
        from core.data_logger import DataLogger
        from api.websocket_server import WebSocketServer
        from api.export_service import ExportService
        from api.routes import APIRoutes
        
        # Create session
        session_id = generate_session_id()
        session_dir = create_session_directory(session_id)
        logger.info(f"Created new session: {session_id} in {session_dir}")
        
        # Initialize components
        camera_manager = CameraManager(session_dir)
        camera_manager.add_participant("participant1", Config.CAMERA_DEVICE_IDS[0])
        camera_manager.add_participant("participant2", Config.CAMERA_DEVICE_IDS[1])
        
        audio_manager = AudioManager(session_dir)
        llm_analyzer = LLMAnalyzer(session_dir)
        
        detector = MisalignmentDetector(
            camera_manager,
            audio_manager,
            llm_analyzer,
            session_dir
        )
        
        data_logger = DataLogger(session_dir, detector)
        data_logger.set_participants(["participant1", "participant2"])
        
        export_service = ExportService(session_dir, data_logger)
        
        websocket_server = WebSocketServer(detector)
        
        # Create and start API routes
        api_routes = APIRoutes(
            detector=detector,
            export_service=export_service,
            port=8080
        )
        
        # Start all components
        logger.info("Starting system components...")
        camera_manager.start_all_cameras()
        audio_manager.start_recording()
        llm_analyzer.start()
        detector.start()
        data_logger.start()
        websocket_server.start()
        api_routes.start()
        
        logger.info("System started in headless mode")
        logger.info(f"WebSocket server running at ws://{Config.WEBSOCKET_HOST}:{Config.WEBSOCKET_PORT}")
        logger.info("API server running at http://localhost:8080")
        
        # Keep running until interrupted
        logger.info("Press Ctrl+C to stop...")
        try:
            # Keep main thread alive
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        
        # Stop components
        api_routes.stop()
        websocket_server.stop()
        data_logger.stop()
        detector.stop()
        llm_analyzer.stop()
        audio_manager.stop_recording()
        camera_manager.stop_all_cameras()
        
        logger.info("System shutdown complete")
    
    except Exception as e:
        log_exception(logger, e, "Error in headless mode")
        sys.exit(1)

def main():
    """Main entry point function."""
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Set log level
        import logging
        log_level = getattr(logging, args.log_level)
        logger.setLevel(log_level)
        
        # Load config if provided
        if args.config:
            config_path = args.config
            if os.path.exists(config_path):
                Config.load_from_file(config_path)
                logger.info(f"Loaded config from {config_path}")
            else:
                logger.warning(f"Config file not found: {config_path}")
        
        # Run in headless mode or with UI
        if args.headless:
            run_headless_mode()
        else:
            run_app()
    
    except Exception as e:
        log_exception(logger, e, "Error in main function")
        sys.exit(1)

if __name__ == "__main__":
    main()