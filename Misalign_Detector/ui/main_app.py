"""
Main Tkinter application for the misalignment detection system.
"""
"""
Add these imports to the top of ui/main_app.py
"""
import os
import time
import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
from datetime import datetime
import numpy as np

from utils.config import Config
from utils.error_handler import get_logger, log_exception
from utils.helpers import generate_session_id, create_session_directory

from ui.visualization import (
    VideoPanel, ScoreChart, TranscriptPanel, 
    AnalysisPanel, ParticipantScorePanel
)
from ui.controls import ControlPanel, SettingsPanel, StatusBar

logger = get_logger(__name__)

class MisalignmentApp:
    """Main application for misalignment detection."""
    
    def __init__(self, root):
        """
        Initialize the application.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Misalignment Detection System")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Apply theming
        self._setup_styles()
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create menu
        self._create_menu()
        
        # Create status bar
        self.status_bar = StatusBar(self.root)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create layout
        self._create_layout()
        
        # Initialize system components
        self.detector = None
        self.camera_manager = None
        self.audio_manager = None
        self.llm_analyzer = None
        self.data_logger = None
        self.websocket_server = None
        self.export_service = None
        
        # Session management
        self.session_id = None
        self.session_dir = None
        
        # UI update thread
        self.is_running = False
        self.update_thread = None
        self.ui_update_interval = Config.UI_REFRESH_RATE_MS / 1000.0  # Convert to seconds
        
        # Set initial status
        self.status_bar.set_status("Ready. Click 'Start Detection' to begin.")
        
        # Log application start
        logger.info("Misalignment Detection Application started")
        
    """
Add the following methods to the MisalignmentApp class in ui/main_app.py
"""

    def process_segment(self):
        """Process the current recording segment - analyze, save and send."""
        try:
            if not self.is_running:
                logger.warning("System not running, cannot process segment")
                return False
            
            # Update status
            self.status_bar.set_status("Processing segment...", show_progress=True)
            
            # Get current transcript and features
            transcript = self.audio_manager.get_transcript(max_segments=5)
            facial_features = self.camera_manager.get_latest_features()
            
            # Force an immediate LLM analysis
            logger.info("Forcing immediate analysis of current segment")
            analysis = self.llm_analyzer.analyze_transcript(transcript, facial_features)
            
            # Wait a moment for analysis to complete (this is a bit of a hack)
            max_wait = 10  # seconds
            start_time = time.time()
            
            # Check if we have a new analysis every 0.5 seconds, for up to max_wait seconds
            while time.time() - start_time < max_wait:
                # Force a detection run
                self.detector._run_detection()
                
                # Get latest detection
                detection = self.detector.get_latest_detection()
                
                if detection and "llm_analysis" in detection:
                    # We have an analysis, break out of the loop
                    break
                
                # Wait a bit
                time.sleep(0.5)
            
            # Get the latest detection
            detection = self.detector.get_latest_detection()
            
            if not detection:
                logger.warning("No detection available")
                self.status_bar.set_status("No detection available")
                return False
            
            # Save segment data
            segment_data = {
                "timestamp": time.time(),
                "formatted_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "transcript": transcript,
                "facial_features": facial_features,
                "detection": detection
            }
            
            # Save to a segment-specific file
            segment_dir = os.path.join(self.session_dir, "segments")
            os.makedirs(segment_dir, exist_ok=True)
            
            segment_id = f"segment_{time.strftime('%Y%m%d_%H%M%S')}"
            segment_file = os.path.join(segment_dir, f"{segment_id}.json")
            
            with open(segment_file, 'w') as f:
                json.dump(segment_data, f, indent=2, default=lambda o: str(o) if isinstance(o, (datetime, np.ndarray)) else o)
            
            # Send via WebSocket if available
            if hasattr(self, 'websocket_server') and self.websocket_server:
                data_to_send = {
                    "type": "segment_complete",
                    "segment_id": segment_id,
                    "timestamp": segment_data["timestamp"],
                    "transcript": transcript,
                    "scores": detection.get("combined_scores", {}),
                    "misalignment_detected": detection.get("misalignment_detected", False)
                }
                
                if "llm_analysis" in detection:
                    data_to_send["cause"] = detection["llm_analysis"].get("cause", "")
                    data_to_send["recommendation"] = detection["llm_analysis"].get("recommendation", "")
                
                self.websocket_server.broadcast(data_to_send)
                logger.info(f"Sent segment {segment_id} data via WebSocket")
            
            self.status_bar.set_status(f"Segment {segment_id} processed and saved")
            logger.info(f"Segment {segment_id} processed and saved to {segment_file}")
            
            return True
        
        except Exception as e:
            error_msg = log_exception(logger, e, "Error processing segment")
            self.status_bar.set_status(f"Error: {error_msg}")
            messagebox.showerror("Segment Error", f"Failed to process segment: {error_msg}")
            return False
    
    # Add this to the _setup_styles method
    def _setup_styles(self):
        """Set up custom styles for the application."""
        try:
            self.style = ttk.Style()
            
            # Check if 'clam' theme is available, otherwise use default
            available_themes = self.style.theme_names()
            if 'clam' in available_themes:
                self.style.theme_use('clam')
            
            # Define custom styles
            self.style.configure("TFrame", background="#F5F5F5")
            self.style.configure("TLabel", background="#F5F5F5")
            self.style.configure("TButton", padding=6)
            self.style.configure("Accent.TButton", foreground="white")
            
            # Configure notebook style
            self.style.configure("TNotebook", background="#F5F5F5")
            self.style.configure("TNotebook.Tab", padding=[10, 5])
            
            # Configure borders for visual separation
            self.style.configure("TLabelframe", borderwidth=2)
            self.style.configure("TLabelframe.Label", font=("Arial", 10, "bold"))
        
        except Exception as e:
            log_exception(logger, e, "Error setting up styles")
    
    def _create_menu(self):
        """Create application menu."""
        self.menu = tk.Menu(self.root)
        self.root.config(menu=self.menu)
        
        # File menu
        file_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Session", command=self._new_session)
        file_menu.add_command(label="Export Session", command=self._export_session)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)
        
        # View menu
        view_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Settings", command=self._show_settings)
        
        # Help menu
        help_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    def _create_layout(self):
        """Create the application layout."""
        # Create notebook for main content
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create main tab
        self.main_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.main_tab, text="Detection")
        
        # Create settings tab
        self.settings_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.settings_tab, text="Settings")
        
        # === Main Tab Layout ===
        
        # Top panel: Video feeds and controls
        top_panel = ttk.Frame(self.main_tab)
        top_panel.pack(fill=tk.X, pady=(0, 10))
        
        # Video feeds (left)
        video_frame = ttk.Frame(top_panel)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create video panels
        self.participant1_video = VideoPanel(video_frame, title="Participant 1")
        self.participant1_video.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.participant2_video = VideoPanel(video_frame, title="Participant 2")
        self.participant2_video.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Controls (right)
        controls_frame = ttk.Frame(top_panel)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create control panel with the new send callback
        self.control_panel = ControlPanel(
            controls_frame,
            on_start=self.start_detection,
            on_stop=self.stop_detection,
            on_export=self.export_session,
            on_send=self.process_segment,  # Add the new send callback
            on_manual_analysis=self.manual_analysis
        )
        self.control_panel.pack(fill=tk.BOTH, expand=True)
        
        # Middle panel: Score displays
        middle_panel = ttk.Frame(self.main_tab)
        middle_panel.pack(fill=tk.X, pady=10)
        
        # Participant scores (left)
        scores_frame = ttk.Frame(middle_panel)
        scores_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        self.participant1_score = ParticipantScorePanel(scores_frame, "participant1", "Participant 1")
        self.participant1_score.pack(side=tk.LEFT, padx=(0, 5))
        
        self.participant2_score = ParticipantScorePanel(scores_frame, "participant2", "Participant 2")
        self.participant2_score.pack(side=tk.LEFT, padx=(5, 0))
        
        # Score chart (right)
        chart_frame = ttk.Frame(middle_panel)
        chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.score_chart = ScoreChart(chart_frame)
        self.score_chart.pack(fill=tk.BOTH, expand=True)
        
        # Bottom panel: Transcript and analysis
        bottom_panel = ttk.Frame(self.main_tab)
        bottom_panel.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Transcript (left)
        transcript_frame = ttk.Frame(bottom_panel)
        transcript_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.transcript_panel = TranscriptPanel(transcript_frame)
        self.transcript_panel.pack(fill=tk.BOTH, expand=True)
        
        # Analysis (right)
        analysis_frame = ttk.Frame(bottom_panel)
        analysis_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.analysis_panel = AnalysisPanel(analysis_frame)
        self.analysis_panel.pack(fill=tk.BOTH, expand=True)
        
        # === Settings Tab Layout ===
        self.settings_panel = SettingsPanel(
            self.settings_tab,
            on_settings_change=self._on_settings_change
        )
        self.settings_panel.pack(fill=tk.BOTH, expand=True)
        
    def _new_session(self):
        """Start a new session."""
        if self.is_running:
            messagebox.showinfo(
                "Session Active",
                "Please stop the current detection session before starting a new one."
            )
            return
        
        # Generate new session ID and directory
        self.session_id = generate_session_id()
        self.session_dir = create_session_directory(self.session_id)
        
        # Update status
        self.status_bar.set_status(f"New session created: {self.session_id}")
        
        # Show message
        messagebox.showinfo(
            "New Session",
            f"New session created with ID: {self.session_id}\n\n"
            f"Directory: {self.session_dir}"
        )
    
    def _export_session(self):
        """Export the current session (menu callback)."""
        if not self.session_id:
            messagebox.showinfo(
                "No Session",
                "No active or completed session to export."
            )
            return
        
        self.export_session("json")
    
    def _show_settings(self):
        """Show settings tab."""
        self.notebook.select(self.settings_tab)
    
    def _show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About Misalignment Detection System",
            "Misalignment Detection System\n\n"
            "A real-time system for detecting communication misalignment\n"
            "using facial features, speech transcription, and LLM analysis.\n\n"
            "Version 1.0.0"
        )
    
    def _on_settings_change(self, settings):
        """
        Handle settings changes.
        
        Args:
            settings (dict): Updated settings
        """
        logger.info(f"Settings updated: {settings}")
        
        # Update detector weights if detector exists
        if hasattr(self, 'detector') and self.detector:
            self.detector.weights = {
                "facial": settings["facial_weight"],
                "llm": settings["llm_weight"]
            }
    
    def start_detection(self):
        """Start the misalignment detection."""
        try:
            if self.is_running:
                logger.warning("Detection already running")
                return False
            
            # Create new session if none exists
            if not self.session_id:
                self.session_id = generate_session_id()
                self.session_dir = create_session_directory(self.session_id)
            
            # Update status
            self.status_bar.set_status("Starting detection...", show_progress=True)
            
            # Import components here to avoid circular imports
            from core.camera_manager import CameraManager
            from core.audio_manager import AudioManager
            from core.llm_analyzer import LLMAnalyzer
            from core.detector import MisalignmentDetector
            from core.data_logger import DataLogger
            from api.websocket_server import WebSocketServer
            from api.export_service import ExportService
            
            # Initialize system components
            logger.info("Initializing system components...")
            
            # Get camera device IDs from settings
            settings = self.settings_panel.get_settings()
            Config.CAMERA_DEVICE_IDS = [settings["camera1_device"], settings["camera2_device"]]
            
            # Initialize components
            self.camera_manager = CameraManager(self.session_dir)
            self.camera_manager.add_participant("participant1", Config.CAMERA_DEVICE_IDS[0])
            self.camera_manager.add_participant("participant2", Config.CAMERA_DEVICE_IDS[1])
            
            self.audio_manager = AudioManager(self.session_dir)
            self.llm_analyzer = LLMAnalyzer(self.session_dir)
            
            self.detector = MisalignmentDetector(
                self.camera_manager,
                self.audio_manager,
                self.llm_analyzer,
                self.session_dir
            )
            
            # Update detector weights from settings
            self.detector.weights = {
                "facial": settings["facial_weight"],
                "llm": settings["llm_weight"]
            }
            
            self.data_logger = DataLogger(self.session_dir, self.detector)
            self.data_logger.set_participants(["participant1", "participant2"])
            
            self.export_service = ExportService(self.session_dir, self.data_logger)
            
            self.websocket_server = WebSocketServer(self.detector)
            
            # Start components
            logger.info("Starting system components...")
            
            # Start in sequence
            self.camera_manager.start_all_cameras()
            self.audio_manager.start_recording()
            self.llm_analyzer.start()
            self.detector.start()
            self.data_logger.start()
            self.websocket_server.start()
            
            # Set flag and start UI update thread
            self.is_running = True
            self.update_thread = threading.Thread(target=self._update_ui, daemon=True)
            self.update_thread.start()
            
            # Update status
            self.status_bar.set_status(f"Detection running - Session: {self.session_id}")
            
            logger.info("Detection started successfully")
            return True
        
        except Exception as e:
            error_msg = log_exception(logger, e, "Error starting detection")
            self.status_bar.set_status(f"Error: {error_msg}")
            messagebox.showerror("Start Error", f"Failed to start detection: {error_msg}")
            
            # Attempt to stop any started components
            self._stop_components()
            
            return False
    
    def stop_detection(self):
        """Stop the misalignment detection."""
        try:
            if not self.is_running:
                logger.warning("Detection not running")
                return False
            
            # Update status
            self.status_bar.set_status("Stopping detection...", show_progress=True)
            
            # Stop components
            self._stop_components()
            
            # Reset flag
            self.is_running = False
            
            # Update status
            self.status_bar.set_status(f"Detection stopped - Session: {self.session_id}")
            
            logger.info("Detection stopped successfully")
            return True
        
        except Exception as e:
            error_msg = log_exception(logger, e, "Error stopping detection")
            self.status_bar.set_status(f"Error stopping: {error_msg}")
            messagebox.showerror("Stop Error", f"Failed to stop detection: {error_msg}")
            return False
    
    def _stop_components(self):
        """Stop all system components."""
        # Stop in reverse sequence
        if hasattr(self, 'websocket_server') and self.websocket_server:
            try:
                self.websocket_server.stop()
            except Exception as e:
                log_exception(logger, e, "Error stopping WebSocket server")
        
        if hasattr(self, 'data_logger') and self.data_logger:
            try:
                self.data_logger.stop()
            except Exception as e:
                log_exception(logger, e, "Error stopping data logger")
        
        if hasattr(self, 'detector') and self.detector:
            try:
                self.detector.stop()
            except Exception as e:
                log_exception(logger, e, "Error stopping detector")
        
        if hasattr(self, 'llm_analyzer') and self.llm_analyzer:
            try:
                self.llm_analyzer.stop()
            except Exception as e:
                log_exception(logger, e, "Error stopping LLM analyzer")
        
        if hasattr(self, 'audio_manager') and self.audio_manager:
            try:
                self.audio_manager.stop_recording()
            except Exception as e:
                log_exception(logger, e, "Error stopping audio manager")
        
        if hasattr(self, 'camera_manager') and self.camera_manager:
            try:
                self.camera_manager.stop_all_cameras()
            except Exception as e:
                log_exception(logger, e, "Error stopping camera manager")
    
    def export_session(self, format="json"):
        """
        Export the session.
        
        Args:
            format (str): Export format ("json" or "zip")
            
        Returns:
            str: Path to export file
        """
        if not self.session_id:
            messagebox.showinfo(
                "No Session",
                "No active or completed session to export."
            )
            return None
        
        if not hasattr(self, 'export_service') or not self.export_service:
            # Create export service if it doesn't exist
            from api.export_service import ExportService
            self.export_service = ExportService(self.session_dir, self.data_logger)
        
        # Export session
        export_path = self.export_service.export_session(format=format)
        return export_path
    
    def manual_analysis(self, context):
        """
        Perform manual analysis.
        
        Args:
            context (str): Context for analysis
            
        Returns:
            dict: Analysis results
        """
        if not hasattr(self, 'llm_analyzer') or not self.llm_analyzer:
            messagebox.showinfo(
                "Not Available",
                "LLM Analyzer not available. Please start detection first."
            )
            return None
        
        # Perform analysis
        analysis = self.llm_analyzer.manual_analysis(context)
        
        # Update analysis panel
        self.analysis_panel.update_analysis(analysis)
        
        return analysis
    
    def _update_ui(self):
        """Update UI elements with latest data."""
        while self.is_running:
            try:
                # Get latest detection
                if self.detector:
                    detection = self.detector.get_latest_detection()
                    
                    if detection:
                        # Update score displays
                        scores = detection.get("combined_scores", {})
                        self.score_chart.update_scores(scores, detection.get("timestamp"))
                        
                        # Update individual participant scores
                        self.participant1_score.update_score(scores.get("participant1", 0.0))
                        self.participant2_score.update_score(scores.get("participant2", 0.0))
                        
                        # Update transcript
                        transcript = detection.get("transcript", "")
                        if transcript:
                            # Extract potential highlight words from cause
                            cause = detection.get("llm_analysis", {}).get("cause", "")
                            highlight_words = []
                            
                            if cause:
                                # Simple extraction of potential key terms
                                words = cause.lower().split()
                                # Filter out common words
                                stop_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to", "of", "in", "for", "with", "on", "at", "by", "about", "as", "into", "like", "through", "after", "over", "between", "out", "against", "during"}
                                highlight_words = [word for word in words if len(word) > 3 and word not in stop_words]
                            
                            self.transcript_panel.update_transcript(transcript, detection.get("timestamp"), highlight_words)
                        
                        # Update analysis panel
                        analysis_data = {
                            "misalignment_detected": detection.get("misalignment_detected", False),
                            "scores": scores,
                            "cause": detection.get("llm_analysis", {}).get("cause", ""),
                            "recommendation": detection.get("llm_analysis", {}).get("recommendation", "")
                        }
                        self.analysis_panel.update_analysis(analysis_data)
                
                # Update video frames
                if self.camera_manager:
                    frames = self.camera_manager.get_latest_frames()
                    
                    if "participant1" in frames and frames["participant1"] is not None:
                        self.participant1_video.update_frame(frames["participant1"])
                    
                    if "participant2" in frames and frames["participant2"] is not None:
                        self.participant2_video.update_frame(frames["participant2"])
                
                # Sleep to avoid using too much CPU
                time.sleep(self.ui_update_interval)
            
            except Exception as e:
                log_exception(logger, e, "Error updating UI")
                time.sleep(0.5)
    
    def run(self):
        """Run the application main loop."""
        try:
            # Check if session directory exists
            if not self.session_id:
                self.session_id = generate_session_id()
                self.session_dir = create_session_directory(self.session_id)
                logger.info(f"Created new session: {self.session_id}")
            
            # Run main loop
            self.root.mainloop()
        
        except Exception as e:
            log_exception(logger, e, "Error in application main loop")
        
        finally:
            # Ensure everything is stopped
            if self.is_running:
                self._stop_components()
            
            logger.info("Application shutdown complete")


def run_app():
    """Run the misalignment detection application."""
    root = tk.Tk()
    app = MisalignmentApp(root)
    app.run()


if __name__ == "__main__":
    run_app()