"""
Detector module for combining signals to detect misalignment.
"""
import os
import time
import json
import threading
from threading import Thread, Event
from datetime import datetime

from utils.config import Config
from utils.error_handler import get_logger, log_exception
from utils.helpers import format_timestamp, normalize_score

logger = get_logger(__name__)

class MisalignmentDetector:
    """Combines signals to detect misalignment in conversations."""
    
    def __init__(self, camera_manager, audio_manager, llm_analyzer, session_dir):
        """
        Initialize the misalignment detector.
        
        Args:
            camera_manager: Camera manager instance
            audio_manager: Audio manager instance
            llm_analyzer: LLM analyzer instance
            session_dir (str): Directory to save session data
        """
        self.camera_manager = camera_manager
        self.audio_manager = audio_manager
        self.llm_analyzer = llm_analyzer
        self.session_dir = session_dir
        
        # Create directory for detector outputs
        self.detector_dir = os.path.join(session_dir, "detector")
        os.makedirs(self.detector_dir, exist_ok=True)
        
        # Detection state
        self.is_running = False
        self.detection_thread = None
        self.detection_interval = Config.ANALYSIS_INTERVAL_MS / 1000.0  # Convert to seconds
        
        # For storing detection history
        self.detection_history = []
        self.max_history_size = 100
        
        # Event for signaling new detection
        self.new_detection_event = Event()
        
        # Latest detection results
        self.latest_detection = None
        
        # Detection weights for combining signals
        self.weights = {
            "facial": 0.4,    # Weight for facial confusion signals
            "llm": 0.6        # Weight for LLM analysis signals
        }
        
        logger.info("Initialized Misalignment Detector")
    def save_segment(self, segment_id=None):
        """
        Save the current state as a detection segment.
        
        Args:
            segment_id (str, optional): Segment identifier
            
        Returns:
            str: Path to saved segment file
        """
        try:
            # Create segments directory if it doesn't exist
            segments_dir = os.path.join(self.session_dir, "segments")
            os.makedirs(segments_dir, exist_ok=True)
            
            # Generate segment ID if not provided
            if segment_id is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                segment_id = f"segment_{timestamp}"
            
            # Get latest detection
            detection = self.get_latest_detection()
            
            if not detection:
                logger.warning("No detection available for segment")
                return None
            
            # Create segment data
            segment_data = {
                "segment_id": segment_id,
                "timestamp": time.time(),
                "formatted_time": format_timestamp(),
                "detection": detection,
                "misalignment_summary": self.get_misalignment_summary()
            }
            
            # Save to file
            segment_file = os.path.join(segments_dir, f"{segment_id}.json")
            
            with open(segment_file, 'w') as f:
                json.dump(segment_data, f, indent=2)
            
            logger.info(f"Saved segment to {segment_file}")
            return segment_file
        
        except Exception as e:
            log_exception(logger, e, "Error saving segment")
            return None

    def get_all_segments(self):
        """
        Get all saved segments.
        
        Returns:
            list: List of segment files
        """
        try:
            segments_dir = os.path.join(self.session_dir, "segments")
            
            if not os.path.exists(segments_dir):
                return []
            
            segment_files = [
                os.path.join(segments_dir, f)
                for f in os.listdir(segments_dir)
                if f.endswith('.json')
            ]
            
            return sorted(segment_files)
        
        except Exception as e:
            log_exception(logger, e, "Error getting segments")
            return []
    
    def start(self):
        """Start the detection thread."""
        if self.is_running:
            logger.warning("Misalignment Detector already running")
            return False
        
        self.is_running = True
        self.detection_thread = Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        
        logger.info("Started Misalignment Detector")
        return True
    
    def stop(self):
        """Stop the detection thread."""
        self.is_running = False
        
        if self.detection_thread:
            self.detection_thread.join(timeout=2.0)
        
        logger.info("Stopped Misalignment Detector")
    
    def _detection_loop(self):
        """Main loop for detecting misalignment."""
        last_run_time = time.time()
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Run detection at the specified interval
                if current_time - last_run_time >= self.detection_interval:
                    self._run_detection()
                    last_run_time = current_time
                
                # Sleep to avoid using too much CPU
                time.sleep(min(0.1, self.detection_interval / 2))
            
            except Exception as e:
                log_exception(logger, e, "Error in detection loop")
                time.sleep(0.5)
    
    def _run_detection(self):
        """Run misalignment detection by combining signals."""
        try:
            # Get facial confusion scores
            facial_scores = self.camera_manager.get_confusion_scores()
            
            # Get latest features for LLM context
            facial_features = self.camera_manager.get_latest_features()
            
            # Get latest transcript
            transcript = self.audio_manager.get_transcript(max_segments=5)
            
            # Queue for LLM analysis if we have enough transcript
            if len(transcript.split()) >= 10:
                self.llm_analyzer.analyze_transcript(transcript, facial_features)
            
            # Get LLM misalignment scores
            llm_scores = self.llm_analyzer.get_misalignment_scores()
            
            # Set temporary scores for LLM analyzer
            # (These will be used until LLM analysis is complete)
            self.llm_analyzer.set_temp_scores(facial_scores)
            
            # Get latest LLM analysis
            llm_analysis = self.llm_analyzer.get_latest_analysis()
            
            # Combine signals
            combined_scores = self._combine_signals(facial_scores, llm_scores)
            
            # Construct detection result
            detection = {
                "timestamp": time.time(),
                "formatted_time": format_timestamp(),
                "facial_scores": facial_scores,
                "llm_scores": llm_scores,
                "combined_scores": combined_scores,
                "misalignment_detected": any(score > 0.5 for score in combined_scores.values()),
                "transcript": transcript
            }
            
            # Add LLM analysis if available
            if llm_analysis:
                detection["llm_analysis"] = {
                    "misalignment_detected": llm_analysis.get("misalignment_detected", False),
                    "cause": llm_analysis.get("cause", ""),
                    "recommendation": llm_analysis.get("recommendation", "")
                }
            
            # Update latest detection
            self.latest_detection = detection
            
            # Add to history
            self.detection_history.append(detection)
            
            # Keep history within size limit
            if len(self.detection_history) > self.max_history_size:
                self.detection_history = self.detection_history[-self.max_history_size:]
            
            # Save detection
            self._save_detection(detection)
            
            # Signal new detection
            self.new_detection_event.set()
            self.new_detection_event.clear()
            
        except Exception as e:
            log_exception(logger, e, "Error running detection")
    
    def _combine_signals(self, facial_scores, llm_scores):
        """
        Combine facial and LLM signals.
        
        Args:
            facial_scores (dict): Facial confusion scores for each participant
            llm_scores (dict): LLM misalignment scores for each participant
            
        Returns:
            dict: Combined misalignment scores for each participant
        """
        combined_scores = {}
        
        # Get all participant IDs from both sources
        all_participants = set(facial_scores.keys()) | set(llm_scores.keys())
        
        for participant_id in all_participants:
            facial_score = facial_scores.get(participant_id, 0.0)
            llm_score = llm_scores.get(participant_id, 0.0)
            
            # Weight and combine scores
            weighted_score = (
                self.weights["facial"] * facial_score +
                self.weights["llm"] * llm_score
            )
            
            # Normalize to 0-1 range
            combined_scores[participant_id] = normalize_score(weighted_score)
        
        return combined_scores
    
    def _save_detection(self, detection):
        """
        Save detection to file.
        
        Args:
            detection (dict): Detection results
        """
        try:
            # Save to CSV for time series data
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(detection["timestamp"]))
            
            # Save detailed JSON every 10 detections
            if len(self.detection_history) % 10 == 0:
                file_path = os.path.join(self.detector_dir, f"detection_{timestamp}.json")
                
                with open(file_path, 'w') as f:
                    json.dump(detection, f, indent=2)
        
        except Exception as e:
            log_exception(logger, e, "Error saving detection")
    
    def get_latest_detection(self):
        """
        Get the latest detection.
        
        Returns:
            dict: Latest detection results
        """
        return self.latest_detection
    
    def get_detection_history(self):
        """
        Get detection history.
        
        Returns:
            list: List of detection results
        """
        return self.detection_history
    
    def wait_for_new_detection(self, timeout=None):
        """
        Wait for a new detection.
        
        Args:
            timeout (float, optional): Maximum time to wait in seconds
            
        Returns:
            bool: True if new detection is available, False if timeout
        """
        return self.new_detection_event.wait(timeout)
    
    def get_misalignment_summary(self):
        """
        Get a summary of misalignment.
        
        Returns:
            dict: Misalignment summary
        """
        if not self.latest_detection:
            return {
                "misalignment_detected": False,
                "scores": {},
                "cause": "",
                "recommendation": ""
            }
        
        summary = {
            "misalignment_detected": self.latest_detection.get("misalignment_detected", False),
            "scores": self.latest_detection.get("combined_scores", {}),
            "cause": "",
            "recommendation": ""
        }
        
        # Add LLM analysis if available
        if "llm_analysis" in self.latest_detection:
            summary["cause"] = self.latest_detection["llm_analysis"].get("cause", "")
            summary["recommendation"] = self.latest_detection["llm_analysis"].get("recommendation", "")
        
        return summary