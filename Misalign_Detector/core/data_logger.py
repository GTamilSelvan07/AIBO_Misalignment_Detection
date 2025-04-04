"""
Data logger module for logging session data in CSV and JSON formats.
"""
import os
import csv
import json
import time
from datetime import datetime
from threading import Thread, Lock, Event

from utils.config import Config
from utils.error_handler import get_logger, log_exception
from utils.helpers import format_timestamp

logger = get_logger(__name__)

class DataLogger:
    """Logs session data in CSV and JSON formats."""
    
    def __init__(self, session_dir, detector):
        """
        Initialize the data logger.
        
        Args:
            session_dir (str): Directory to save session data
            detector: Misalignment detector instance
        """
        self.session_dir = session_dir
        self.detector = detector
        
        # Create logging directories
        self.logs_dir = os.path.join(session_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Define log file paths
        self.csv_log_path = os.path.join(self.logs_dir, "misalignment_scores.csv")
        self.json_log_path = os.path.join(self.logs_dir, "session_log.json")
        self.session_export_path = os.path.join(self.logs_dir, "session_export.json")
        
        # Logger state
        self.is_logging = False
        self.logging_thread = None
        self.logging_interval = 1.0  # seconds
        
        # Event to signal stop
        self.stop_event = Event()
        
        # Lock for thread safety
        self.lock = Lock()
        
        # Session metadata
        self.session_metadata = {
            "session_id": os.path.basename(session_dir),
            "start_time": format_timestamp(),
            "participants": [],
            "end_time": None,
            "total_duration": 0,
            "misalignment_summary": {
                "average_score": 0,
                "peak_score": 0,
                "duration_above_threshold": 0
            }
        }
        
        # Initialize CSV log file with headers
        self._initialize_csv_log()
        
        logger.info("Initialized DataLogger")
    
    def _initialize_csv_log(self):
        """Initialize CSV log file with headers."""
        try:
            if not os.path.exists(self.csv_log_path):
                with open(self.csv_log_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    headers = [
                        "timestamp", 
                        "unix_timestamp", 
                        "misalignment_detected"
                    ]
                    
                    # Add placeholder for participant scores
                    # (We'll update with actual participants when logging starts)
                    headers.extend(["participant1_score", "participant2_score"])
                    
                    writer.writerow(headers)
        
        except Exception as e:
            log_exception(logger, e, "Error initializing CSV log")
    
    def set_participants(self, participants):
        """
        Set participant information.
        
        Args:
            participants (list): List of participant IDs
        """
        with self.lock:
            self.session_metadata["participants"] = participants
            
            # Update CSV headers with actual participant IDs
            if os.path.exists(self.csv_log_path):
                with open(self.csv_log_path, 'r', newline='') as f:
                    reader = csv.reader(f)
                    headers = next(reader)
                
                # Create new headers with actual participant IDs
                new_headers = headers[:3]  # Keep timestamp, unix_timestamp, misalignment_detected
                for participant in participants:
                    new_headers.append(f"{participant}_score")
                
                # Create new CSV with updated headers
                temp_path = self.csv_log_path + ".temp"
                with open(temp_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(new_headers)
                
                # Replace original CSV
                if os.path.exists(self.csv_log_path):
                    os.remove(self.csv_log_path)
                os.rename(temp_path, self.csv_log_path)
    
    def start(self):
        """Start the logging thread."""
        if self.is_logging:
            logger.warning("DataLogger already running")
            return False
        
        self.is_logging = True
        self.stop_event.clear()
        self.logging_thread = Thread(target=self._logging_loop, daemon=True)
        self.logging_thread.start()
        
        logger.info("Started DataLogger")
        return True
    
    def stop(self):
        """Stop the logging thread."""
        if not self.is_logging:
            return
        
        self.is_logging = False
        self.stop_event.set()
        
        if self.logging_thread:
            self.logging_thread.join(timeout=2.0)
        
        # Update session metadata
        with self.lock:
            self.session_metadata["end_time"] = format_timestamp()
            
            # Calculate session duration
            start_time = datetime.strptime(self.session_metadata["start_time"], "%Y-%m-%d %H:%M:%S.%f")
            end_time = datetime.strptime(self.session_metadata["end_time"], "%Y-%m-%d %H:%M:%S.%f")
            duration_seconds = (end_time - start_time).total_seconds()
            self.session_metadata["total_duration"] = duration_seconds
        
        # Generate final session export
        self.export_session()
        
        logger.info("Stopped DataLogger")
    
    def _logging_loop(self):
        """Main loop for logging data."""
        last_log_time = time.time()
        
        while self.is_logging and not self.stop_event.is_set():
            try:
                current_time = time.time()
                
                # Log at the specified interval
                if current_time - last_log_time >= self.logging_interval:
                    self._log_current_state()
                    last_log_time = current_time
                
                # Sleep to avoid using too much CPU
                time.sleep(min(0.1, self.logging_interval / 2))
            
            except Exception as e:
                log_exception(logger, e, "Error in logging loop")
                time.sleep(0.5)
    
    def _log_current_state(self):
        """Log the current state to CSV and JSON."""
        try:
            # Get latest detection
            detection = self.detector.get_latest_detection()
            if not detection:
                return
            
            # Log to CSV
            self._log_to_csv(detection)
            
            # Log to JSON (less frequently)
            if time.time() % 10 < 1:  # Log every ~10 seconds
                self._log_to_json(detection)
            
        except Exception as e:
            log_exception(logger, e, "Error logging current state")
    
    def _log_to_csv(self, detection):
        """
        Log detection to CSV.
        
        Args:
            detection (dict): Detection results
        """
        try:
            timestamp = format_timestamp(detection["timestamp"])
            unix_timestamp = detection["timestamp"]
            misalignment_detected = 1 if detection.get("misalignment_detected", False) else 0
            
            # Get participant scores
            combined_scores = detection.get("combined_scores", {})
            
            # Prepare row data
            row = [timestamp, unix_timestamp, misalignment_detected]
            
            # Add scores for each participant
            with self.lock:
                for participant in self.session_metadata["participants"]:
                    row.append(combined_scores.get(participant, 0.0))
            
            # Write to CSV
            with open(self.csv_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        
        except Exception as e:
            log_exception(logger, e, "Error logging to CSV")
    
    def _log_to_json(self, detection):
        """
        Log detection to JSON.
        
        Args:
            detection (dict): Detection results
        """
        try:
            # Read existing log if it exists
            if os.path.exists(self.json_log_path):
                with open(self.json_log_path, 'r') as f:
                    log_data = json.load(f)
            else:
                log_data = {
                    "session_id": os.path.basename(self.session_dir),
                    "start_time": self.session_metadata["start_time"],
                    "detections": []
                }
            
            # Add detection with minimal data to conserve file size
            log_data["detections"].append({
                "timestamp": detection["timestamp"],
                "misalignment_detected": detection.get("misalignment_detected", False),
                "combined_scores": detection.get("combined_scores", {}),
                "cause": detection.get("llm_analysis", {}).get("cause", "")
            })
            
            # Limit the number of detections in the log
            if len(log_data["detections"]) > 100:
                log_data["detections"] = log_data["detections"][-100:]
            
            # Write to JSON
            with open(self.json_log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
        
        except Exception as e:
            log_exception(logger, e, "Error logging to JSON")
    
    def export_session(self):
        """
        Export session data to a comprehensive JSON file.
        
        Returns:
            str: Path to export file
        """
        try:
            # Create export data structure
            export_data = {
                "metadata": self.session_metadata,
                "logs": {
                    "csv_path": os.path.relpath(self.csv_log_path, self.session_dir),
                    "json_path": os.path.relpath(self.json_log_path, self.session_dir)
                },
                "files": {
                    "video": [],
                    "audio": [],
                    "transcripts": [],
                    "features": [],
                    "analysis": []
                },
                "summary": {
                    "misalignment_score_history": [],
                    "transcript_highlights": []
                }
            }
            
            # List video files
            video_dir = os.path.join(self.session_dir, "video")
            if os.path.exists(video_dir):
                export_data["files"]["video"] = [
                    os.path.join("video", f) for f in os.listdir(video_dir) if f.endswith(('.avi', '.mp4'))
                ]
            
            # List audio files
            audio_dir = os.path.join(self.session_dir, "audio")
            if os.path.exists(audio_dir):
                export_data["files"]["audio"] = [
                    os.path.join("audio", f) for f in os.listdir(audio_dir) if f.endswith('.wav')
                ]
            
            # List transcript files
            transcript_dir = os.path.join(self.session_dir, "transcripts")
            if os.path.exists(transcript_dir):
                export_data["files"]["transcripts"] = [
                    os.path.join("transcripts", f) for f in os.listdir(transcript_dir) if f.endswith('.json')
                ]
            
            # List feature files
            features_dir = os.path.join(self.session_dir, "features")
            if os.path.exists(features_dir):
                for participant in os.listdir(features_dir):
                    participant_dir = os.path.join(features_dir, participant)
                    if os.path.isdir(participant_dir):
                        export_data["files"]["features"].extend([
                            os.path.join("features", participant, f) 
                            for f in os.listdir(participant_dir) 
                            if f.endswith('.csv') and os.path.isfile(os.path.join(participant_dir, f))
                        ])
            
            # List analysis files
            analysis_dir = os.path.join(self.session_dir, "analysis")
            if os.path.exists(analysis_dir):
                export_data["files"]["analysis"] = [
                    os.path.join("analysis", f) for f in os.listdir(analysis_dir) if f.endswith('.json')
                ]
            
            # Generate misalignment score history
            if os.path.exists(self.csv_log_path):
                with open(self.csv_log_path, 'r', newline='') as f:
                    reader = csv.reader(f)
                    headers = next(reader)
                    
                    # Find indices of score columns
                    score_indices = [i for i, header in enumerate(headers) if "_score" in header]
                    
                    # Extract scores from each row
                    for row in reader:
                        if len(row) > max(score_indices):
                            export_data["summary"]["misalignment_score_history"].append({
                                "timestamp": row[0],
                                "scores": {
                                    headers[i].replace("_score", ""): float(row[i])
                                    for i in score_indices if i < len(row) and row[i]
                                }
                            })
            
            # Get transcript highlights (moments with high misalignment)
            detection_history = self.detector.get_detection_history()
            high_misalignment_detections = [
                d for d in detection_history
                if d.get("misalignment_detected", False) and "transcript" in d
            ]
            
            for detection in high_misalignment_detections:
                highlight = {
                    "timestamp": format_timestamp(detection["timestamp"]),
                    "transcript": detection.get("transcript", ""),
                    "cause": detection.get("llm_analysis", {}).get("cause", "")
                }
                export_data["summary"]["transcript_highlights"].append(highlight)
            
            # Calculate summary statistics
            scores_history = export_data["summary"]["misalignment_score_history"]
            if scores_history:
                all_scores = []
                for entry in scores_history:
                    all_scores.extend(entry["scores"].values())
                
                if all_scores:
                    avg_score = sum(all_scores) / len(all_scores)
                    peak_score = max(all_scores)
                    above_threshold = len([s for s in all_scores if s > 0.5])
                    duration_above_threshold = above_threshold * self.logging_interval
                    
                    export_data["metadata"]["misalignment_summary"] = {
                        "average_score": avg_score,
                        "peak_score": peak_score,
                        "duration_above_threshold": duration_above_threshold
                    }
            
            # Write to export file
            with open(self.session_export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported session data to {self.session_export_path}")
            return self.session_export_path
        
        except Exception as e:
            error_msg = log_exception(logger, e, "Error exporting session data")
            # Create minimal export if error occurred
            with open(self.session_export_path, 'w') as f:
                json.dump({
                    "metadata": self.session_metadata,
                    "error": error_msg
                }, f, indent=2)
            return self.session_export_path