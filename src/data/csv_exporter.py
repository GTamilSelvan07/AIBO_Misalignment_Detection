"""
CSV exporter for misalignment predictions.
"""
import os
import time
import threading
import queue
import csv
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger

from config import config, LOGS_DIR


class CSVExporter:
    """
    Exports misalignment predictions to CSV files.
    """
    def __init__(self, export_dir: Optional[Path] = None, export_interval: float = 60.0):
        """
        Initialize the CSV exporter.
        
        Args:
            export_dir: Directory to export CSV files to
            export_interval: Interval between exports in seconds
        """
        self.export_dir = export_dir or LOGS_DIR / "exports"
        os.makedirs(self.export_dir, exist_ok=True)
        
        self.export_interval = export_interval
        self.last_export_time = time.time()
        
        # Data to be exported
        self.prediction_data = []
        
        # Export queue for asynchronous processing
        self.export_queue = queue.Queue()
        self.is_exporting = False
        self.exporting_thread = None
        
        # Start the export thread
        self.start_exporting()
        
    def start_exporting(self):
        """
        Start the export thread.
        """
        if self.is_exporting:
            logger.warning("CSV exporting already running")
            return
            
        self.is_exporting = True
        self.exporting_thread = threading.Thread(target=self._exporting_loop, daemon=True)
        self.exporting_thread.start()
        logger.info("Started CSV export thread")
        
    def _exporting_loop(self):
        """
        Main loop for exporting data to CSV.
        """
        while self.is_exporting:
            try:
                # Check if it's time to export
                current_time = time.time()
                if current_time - self.last_export_time >= self.export_interval:
                    # Export the data
                    self._export_data()
                    self.last_export_time = current_time
                
                # Also check the export queue for manual exports
                try:
                    export_item = self.export_queue.get(timeout=1.0)
                    self._handle_export_item(export_item)
                    self.export_queue.task_done()
                except queue.Empty:
                    # No items in queue, just continue
                    pass
                
                # Sleep to prevent tight loop
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error in CSV export loop: {str(e)}")
    
    def _handle_export_item(self, export_item: Dict):
        """
        Handle an export item from the queue.
        
        Args:
            export_item: Export item with 'type' and other fields
        """
        if export_item['type'] == 'prediction':
            # Add prediction data
            self.add_prediction(
                export_item['person_name'],
                export_item['timestamp'],
                export_item['combined_score'],
                export_item['camera_score'],
                export_item['speech_score'],
                export_item['details']
            )
            
        elif export_item['type'] == 'manual_export':
            # Manual export request
            self._export_data(export_item.get('filename', None))
    
    def add_prediction(self, person_name: str, timestamp: float, 
                      combined_score: int, camera_score: int, speech_score: int,
                      details: Optional[Dict] = None):
        """
        Add a prediction to the export data.
        
        Args:
            person_name: Identifier for the person
            timestamp: Time of the prediction
            combined_score: Combined misalignment score
            camera_score: Camera misalignment score
            speech_score: Speech misalignment score
            details: Optional additional details
        """
        # Extract additional data from details
        active_aus = []
        speech_text = ""
        speech_indicators = []
        llm_analysis = None
        
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
            
            # LLM analysis from combined score
            if "llm_analysis" in details:
                llm_analysis = details["llm_analysis"]
        
        # Create timestamp string
        timestamp_str = datetime.fromtimestamp(timestamp).isoformat()
        
        # Create prediction data
        prediction = {
            "timestamp": timestamp_str,
            "person_name": person_name,
            "combined_score": combined_score,
            "camera_score": camera_score,
            "speech_score": speech_score,
            "active_aus": ",".join([f"AU{au}" for au in active_aus]),
            "speech_text": speech_text,
            "speech_indicators": ",".join(speech_indicators),
        }
        
        # Add LLM analysis if available
        if llm_analysis:
            if isinstance(llm_analysis, dict):
                # Add relevant fields
                for key in ["explanation", "likely_cause", "confidence"]:
                    if key in llm_analysis:
                        prediction[f"llm_{key}"] = llm_analysis[key]
                
                # Handle list fields
                for key in ["indicators", "suggestions"]:
                    if key in llm_analysis and isinstance(llm_analysis[key], list):
                        prediction[f"llm_{key}"] = ",".join(llm_analysis[key])
        
        # Add to prediction data
        self.prediction_data.append(prediction)
    
    def queue_prediction(self, person_name: str, timestamp: float, 
                        combined_score: int, camera_score: int, speech_score: int,
                        details: Optional[Dict] = None):
        """
        Queue a prediction for export.
        
        Args:
            person_name: Identifier for the person
            timestamp: Time of the prediction
            combined_score: Combined misalignment score
            camera_score: Camera misalignment score
            speech_score: Speech misalignment score
            details: Optional additional details
        """
        # Create export item
        export_item = {
            'type': 'prediction',
            'person_name': person_name,
            'timestamp': timestamp,
            'combined_score': combined_score,
            'camera_score': camera_score,
            'speech_score': speech_score,
            'details': details
        }
        
        # Add to export queue
        self.export_queue.put(export_item)
    
    def manual_export(self, filename: Optional[str] = None):
        """
        Manually trigger an export.
        
        Args:
            filename: Optional filename to use for the export
        """
        # Create export item
        export_item = {
            'type': 'manual_export',
            'filename': filename
        }
        
        # Add to export queue
        self.export_queue.put(export_item)
    
    def _export_data(self, filename: Optional[str] = None):
        """
        Export the prediction data to a CSV file.
        
        Args:
            filename: Optional filename to use for the export
        """
        if not self.prediction_data:
            logger.debug("No prediction data to export")
            return
        
        try:
            # Generate filename if not provided
            if filename is None:
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"misalignment_predictions_{current_time}.csv"
            
            # Full path to the export file
            export_path = os.path.join(self.export_dir, filename)
            
            # Convert to DataFrame for easier handling
            df = pd.DataFrame(self.prediction_data)
            
            # Write to CSV
            df.to_csv(export_path, index=False)
            
            logger.info(f"Exported {len(self.prediction_data)} predictions to {export_path}")
            
            # Clear the prediction data after export
            self.prediction_data = []
            
        except Exception as e:
            logger.error(f"Error exporting prediction data: {str(e)}")
    
    def stop_exporting(self):
        """
        Stop the export thread.
        """
        # Export any remaining data before stopping
        if self.prediction_data:
            self._export_data()
            
        self.is_exporting = False
        
        # Wait for export thread to end
        if self.exporting_thread is not None and self.exporting_thread.is_alive():
            self.exporting_thread.join(timeout=1.0)
            
        logger.info("Stopped CSV export thread")
    
    def set_export_interval(self, interval: float):
        """
        Set the export interval.
        
        Args:
            interval: Export interval in seconds
        """
        self.export_interval = max(10.0, interval)
        logger.info(f"Set CSV export interval to {self.export_interval} seconds")