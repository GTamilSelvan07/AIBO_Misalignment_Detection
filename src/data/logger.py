"""
Logging functionality for the misalignment detection system.
"""
import os
import time
import threading
import csv
import queue
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, IO
from loguru import logger
import sys

from config import config, LOGS_DIR


def setup_logging():
    """
    Set up the application-wide logging configuration.
    """
    # Remove default logger
    logger.remove()
    
    # Set up log format
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    
    # Add console logger if enabled
    if config.logging.console_logging:
        logger.add(
            sys.stderr,
            format=log_format,
            level=config.logging.log_level,
            colorize=True
        )
        
    # Add file logger if enabled
    if config.logging.file_logging:
        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(config.logging.log_file), exist_ok=True)
        
        logger.add(
            config.logging.log_file,
            format=log_format,
            level=config.logging.log_level,
            rotation=config.logging.log_rotation,
            compression="zip"
        )
        
    logger.info("Logging initialized")
    

class MisalignmentLogger:
    """
    Logger for misalignment detection results.
    """
    def __init__(self, log_interval: float = None):
        """
        Initialize the misalignment logger.
        
        Args:
            log_interval: Interval between log entries (seconds)
        """
        self.log_interval = log_interval or config.logging.score_log_interval
        
        # Scores directory
        self.scores_dir = LOGS_DIR / "misalignment_scores"
        os.makedirs(self.scores_dir, exist_ok=True)
        
        # CSV files for each person
        self.csv_files = {}  # {person_name: file_object}
        self.csv_writers = {}  # {person_name: csv_writer}
        
        # Last log time for each person
        self.last_log_time = {}  # {person_name: timestamp}
        
        # Async logging
        self.log_queue = queue.Queue()
        self.is_logging = False
        self.logging_thread = None
        
    def start_logging(self):
        """
        Start the asynchronous logging thread.
        """
        if self.is_logging:
            logger.warning("Misalignment logging already running")
            return
            
        self.is_logging = True
        self.logging_thread = threading.Thread(target=self._logging_loop, daemon=True)
        self.logging_thread.start()
        logger.info("Started misalignment logging")
        
    def _logging_loop(self):
        """
        Loop for asynchronous logging.
        """
        while self.is_logging:
            try:
                # Get the next item to log
                log_item = self.log_queue.get(timeout=1.0)
                
                # Log the item
                self._log_item(*log_item)
                
                # Mark task as done
                self.log_queue.task_done()
                
            except queue.Empty:
                # No items to log, just continue
                pass
                
            except Exception as e:
                logger.error(f"Error in misalignment logging loop: {str(e)}")
                
    def _log_item(self, person_name: str, timestamp: float, 
                combined_score: int, camera_score: int, speech_score: int,
                details: Optional[Dict] = None):
        """
        Log a misalignment item to CSV.
        
        Args:
            person_name: Identifier for the person
            timestamp: Time of the score
            combined_score: Combined misalignment score
            camera_score: Camera misalignment score
            speech_score: Speech misalignment score
            details: Optional additional details
        """
        try:
            # Ensure CSV file is open for this person
            if person_name not in self.csv_files:
                self._create_csv_file(person_name)
                
            # Format timestamp
            dt = datetime.fromtimestamp(timestamp)
            timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            
            # Extract additional data from details
            active_aus = []
            speech_text = ""
            speech_indicators = []
            
            if details:
                # Camera details
                camera_details = details.get("camera_details", {})
                if camera_details and "active_aus" in camera_details:
                    active_aus = list(camera_details["active_aus"].keys())
                    
                # Speech details
                speech_text = details.get("speech_text", "")
                speech_details = details.get("speech_details", {})
                if speech_details and "llm_analysis" in speech_details:
                    llm_analysis = speech_details["llm_analysis"]
                    if llm_analysis and "indicators" in llm_analysis:
                        speech_indicators = llm_analysis["indicators"]
                        
            # Prepare row data
            row = [
                timestamp_str,
                combined_score,
                camera_score,
                speech_score,
                ",".join([f"AU{au}" for au in active_aus]),
                speech_text,
                ",".join(speech_indicators)
            ]
            
            # Write to CSV
            self.csv_writers[person_name].writerow(row)
            self.csv_files[person_name].flush()
            
        except Exception as e:
            logger.error(f"Error logging misalignment for {person_name}: {str(e)}")
            
    def _create_csv_file(self, person_name: str):
        """
        Create a new CSV file for a person.
        
        Args:
            person_name: Identifier for the person
        """
        # Close existing file if open
        if person_name in self.csv_files:
            self.csv_files[person_name].close()
            
        # Create filename with date
        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"{person_name}_{date_str}.csv"
        filepath = os.path.join(self.scores_dir, filename)
        
        # Check if file exists
        file_exists = os.path.exists(filepath)
        
        # Open file
        self.csv_files[person_name] = open(filepath, 'a', newline='')
        self.csv_writers[person_name] = csv.writer(self.csv_files[person_name])
        
        # Write header if new file
        if not file_exists:
            self.csv_writers[person_name].writerow([
                "Timestamp",
                "Combined_Score",
                "Camera_Score",
                "Speech_Score",
                "Active_AUs",
                "Speech_Text",
                "Speech_Indicators"
            ])
            
        logger.info(f"Created CSV log file for {person_name}: {filepath}")
        
    def log_scores(self, person_name: str, timestamp: float, 
                  combined_score: int, camera_score: int, speech_score: int,
                  details: Optional[Dict] = None, force: bool = False):
        """
        Log misalignment scores for a person.
        
        Args:
            person_name: Identifier for the person
            timestamp: Time of the score
            combined_score: Combined misalignment score
            camera_score: Camera misalignment score
            speech_score: Speech misalignment score
            details: Optional additional details
            force: Force logging regardless of interval
        """
        # Check if it's time to log
        last_time = self.last_log_time.get(person_name, 0)
        if not force and timestamp - last_time < self.log_interval:
            return
            
        # Update last log time
        self.last_log_time[person_name] = timestamp
        
        # Queue for async logging
        if not self.is_logging:
            self.start_logging()
            
        self.log_queue.put((person_name, timestamp, combined_score, camera_score, speech_score, details))
        
    def stop_logging(self):
        """
        Stop logging and close all files.
        """
        self.is_logging = False
        
        # Wait for logging thread to end
        if self.logging_thread is not None and self.logging_thread.is_alive():
            self.logging_thread.join(timeout=1.0)
            
        # Close all files
        for file in self.csv_files.values():
            file.close()
            
        self.csv_files = {}
        self.csv_writers = {}
        
        logger.info("Stopped misalignment logging")
        
    def get_log_files(self) -> Dict[str, List[str]]:
        """
        Get a list of log files for each person.
        
        Returns:
            dict: {person_name: [filepath1, filepath2, ...]}
        """
        result = {}
        
        try:
            # List all files in the scores directory
            files = os.listdir(self.scores_dir)
            
            for filename in files:
                if filename.endswith(".csv"):
                    # Extract person name from filename
                    parts = filename.split("_")
                    if len(parts) >= 2:
                        person_name = parts[0]
                        
                        if person_name not in result:
                            result[person_name] = []
                            
                        result[person_name].append(os.path.join(self.scores_dir, filename))
                        
        except Exception as e:
            logger.error(f"Error getting log files: {str(e)}")
            
        return result
        
    def set_log_interval(self, interval: float):
        """
        Set the logging interval.
        
        Args:
            interval: Logging interval in seconds
        """
        self.log_interval = max(0.1, interval)
        logger.info(f"Set misalignment logging interval to {self.log_interval} seconds")