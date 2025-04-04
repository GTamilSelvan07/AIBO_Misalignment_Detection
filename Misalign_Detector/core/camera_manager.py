"""
Camera manager module for capturing video and extracting facial features.
"""
import os
import cv2
import time
import threading
import subprocess
import numpy as np
import pandas as pd
from queue import Queue
from threading import Thread
from pathlib import Path

from utils.config import Config
from utils.error_handler import get_logger, CameraError, log_exception
from utils.helpers import format_timestamp, extract_emotions_from_aus

logger = get_logger(__name__)

class ParticipantCamera:
    """Manages camera capture and feature extraction for a single participant."""
    
    def __init__(self, device_id, participant_id, session_dir, frame_queue=None):
        """
        Initialize a camera for a participant.
        
        Args:
            device_id (int): Camera device ID
            participant_id (str): Identifier for the participant (e.g., "participant1")
            session_dir (str): Directory to save session data
            frame_queue (Queue, optional): Queue to send frames to for processing
        """
        self.device_id = device_id
        self.participant_id = participant_id
        self.session_dir = session_dir
        self.frame_queue = frame_queue or Queue(maxsize=30)
        
        self.cap = None
        self.is_running = False
        self.capture_thread = None
        self.current_frame = None
        self.frame_count = 0
        
        # Feature extraction
        self.features_queue = Queue(maxsize=10)
        self.latest_features = {}
        self.feature_extraction_thread = None
        
        # Create directories for saving data
        self.video_dir = os.path.join(session_dir, "video")
        self.features_dir = os.path.join(session_dir, "features", participant_id)
        os.makedirs(self.features_dir, exist_ok=True)
        
        # Paths for video and feature files
        self.video_path = os.path.join(self.video_dir, f"{participant_id}.avi")
        self.features_csv_path = os.path.join(self.features_dir, f"{participant_id}_features.csv")
        
        # OpenFace paths
        self.openface_input_dir = os.path.join(self.features_dir, "input")
        self.openface_output_dir = os.path.join(self.features_dir, "output")
        os.makedirs(self.openface_input_dir, exist_ok=True)
        os.makedirs(self.openface_output_dir, exist_ok=True)
        
        logger.info(f"Initialized camera for {participant_id} with device ID {device_id}")
    
    def start(self):
        """Start camera capture and feature extraction threads."""
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            if not self.cap.isOpened():
                raise CameraError(f"Failed to open camera {self.device_id} for {self.participant_id}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
            
            # Get actual camera properties (might differ from requested)
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(
                self.video_path, fourcc, actual_fps, (actual_width, actual_height)
            )
            
            logger.info(f"Started camera for {self.participant_id}: {actual_width}x{actual_height} at {actual_fps} FPS")
            
            self.is_running = True
            self.capture_thread = Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            self.feature_extraction_thread = Thread(target=self._feature_extraction_loop, daemon=True)
            self.feature_extraction_thread.start()
            
            return True
        
        except Exception as e:
            error_msg = log_exception(logger, e, f"Failed to start camera for {self.participant_id}")
            raise CameraError(error_msg)
    
    def stop(self):
        """Stop camera capture and feature extraction."""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        
        if self.feature_extraction_thread:
            self.feature_extraction_thread.join(timeout=1.0)
        
        if self.cap:
            self.cap.release()
        
        if hasattr(self, 'video_writer'):
            self.video_writer.release()
        
        logger.info(f"Stopped camera for {self.participant_id}")
    
    def _capture_loop(self):
        """Main loop for capturing frames from the camera."""
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame from camera for {self.participant_id}")
                    time.sleep(0.1)
                    continue
                
                timestamp = time.time()
                self.current_frame = frame.copy()
                self.frame_count += 1
                
                # Save frame to video file
                if hasattr(self, 'video_writer'):
                    self.video_writer.write(frame)
                
                # Save frame periodically for OpenFace processing
                if self.frame_count % (Config.CAMERA_FPS // 2) == 0:  # Process 2 frames per second
                    frame_filename = os.path.join(
                        self.openface_input_dir, 
                        f"{self.participant_id}_{format_timestamp(timestamp).replace(' ', '_').replace(':', '-')}.jpg"
                    )
                    cv2.imwrite(frame_filename, frame)
                    self.frame_queue.put((self.participant_id, frame, timestamp, frame_filename))
                
                # Add slight delay to reduce CPU usage
                time.sleep(1.0 / Config.CAMERA_FPS)
            
            except Exception as e:
                log_exception(logger, e, f"Error in capture loop for {self.participant_id}")
                time.sleep(0.5)
    
    def _feature_extraction_loop(self):
        """
        Loop to process frames with OpenFace and extract features.
        This runs periodically to batch process saved frames.
        """
        last_process_time = time.time()
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Process OpenFace batch every 3 seconds
                if current_time - last_process_time >= 3.0 and os.listdir(self.openface_input_dir):
                    self._run_openface_batch()
                    last_process_time = current_time
                
                time.sleep(0.5)
            
            except Exception as e:
                log_exception(logger, e, f"Error in feature extraction loop for {self.participant_id}")
                time.sleep(1.0)
    
    def _run_openface_batch(self):
        """Run OpenFace on the batch of saved frames."""
        try:
            input_dir = self.openface_input_dir
            output_dir = self.openface_output_dir
            
            # Build OpenFace command
            cmd = [
                Config.OPENFACE_EXE,
                "-fdir", input_dir,
                "-out_dir", output_dir,
                *Config.OPENFACE_OPTIONS.split()
            ]
            
            # Run OpenFace as subprocess
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"OpenFace error for {self.participant_id}: {stderr}")
                return
            
            # Process the OpenFace output CSV files
            self._process_openface_output()
            
            # Clean up input directory (keep only the last few frames for debugging)
            input_files = os.listdir(input_dir)
            if len(input_files) > 5:
                for file in sorted(input_files)[:-5]:
                    os.remove(os.path.join(input_dir, file))
            
        except Exception as e:
            log_exception(logger, e, f"Error running OpenFace for {self.participant_id}")
    
    def _process_openface_output(self):
        """Process OpenFace output files to extract features."""
        try:
            csv_files = [f for f in os.listdir(self.openface_output_dir) if f.endswith('.csv')]
            
            for csv_file in csv_files:
                file_path = os.path.join(self.openface_output_dir, csv_file)
                
                # Read the OpenFace CSV output
                try:
                    df = pd.read_csv(file_path)
                    
                    # Extract relevant features (AUs, head pose, gaze)
                    features = {}
                    
                    # Extract Action Units (AU) intensities
                    au_columns = [col for col in df.columns if col.startswith('AU') and '_r' in col]
                    for au_col in au_columns:
                        au_name = au_col.split('_')[0]
                        if not df[au_col].empty:
                            features[au_name] = float(df[au_col].iloc[0])
                    
                    # Extract head pose
                    pose_columns = ['pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz']
                    for pose_col in pose_columns:
                        if pose_col in df.columns and not df[pose_col].empty:
                            features[pose_col] = float(df[pose_col].iloc[0])
                    
                    # Extract gaze direction
                    gaze_columns = ['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z']
                    for gaze_col in gaze_columns:
                        if gaze_col in df.columns and not df[gaze_col].empty:
                            features[gaze_col] = float(df[gaze_col].iloc[0])
                    
                    # Extract emotions using our helper function
                    emotions = extract_emotions_from_aus(features)
                    features.update(emotions)
                    
                    # Timestamp from the filename
                    timestamp_str = csv_file.split('_')[-1].replace('.csv', '')
                    features['timestamp'] = timestamp_str
                    
                    # Update latest features
                    self.latest_features = features
                    
                    # Put features in queue for external processing
                    self.features_queue.put(features)
                    
                    # Append to features CSV
                    self._append_features_to_csv(features)
                    
                except Exception as e:
                    log_exception(logger, e, f"Error processing OpenFace CSV {csv_file}")
                    continue
            
        except Exception as e:
            log_exception(logger, e, "Error processing OpenFace output")
    
    def _append_features_to_csv(self, features):
        """Append extracted features to CSV file."""
        try:
            # Check if file exists to write header
            file_exists = os.path.isfile(self.features_csv_path)
            
            with open(self.features_csv_path, 'a', newline='') as f:
                # Write header if file is new
                if not file_exists:
                    header = ['timestamp'] + sorted(k for k in features.keys() if k != 'timestamp')
                    f.write(','.join(header) + '\n')
                
                # Write feature values
                values = [features.get('timestamp', '')]
                values += [str(features.get(k, '')) for k in sorted(features.keys()) if k != 'timestamp']
                f.write(','.join(values) + '\n')
            
        except Exception as e:
            log_exception(logger, e, "Error appending features to CSV")
    
    def get_latest_features(self):
        """Get the most recent facial features extracted."""
        return self.latest_features
    
    def get_confusion_score(self):
        """Get the confusion score from the latest features."""
        if not self.latest_features:
            return 0.0
        
        # Use the confusion score directly if it was calculated
        if 'confusion' in self.latest_features:
            return min(1.0, max(0.0, self.latest_features['confusion']))
        
        # As a fallback, use a simple heuristic based on AUs associated with confusion
        confusion_aus = ['AU4', 'AU7', 'AU23']
        au_values = [self.latest_features.get(au, 0.0) for au in confusion_aus]
        if not au_values:
            return 0.0
        
        score = sum(au_values) / len(au_values)
        return min(1.0, max(0.0, score))
    
    def get_current_frame(self):
        """Get the most recent camera frame."""
        return self.current_frame


class CameraManager:
    """Manages multiple cameras for multiple participants."""
    
    def __init__(self, session_dir):
        """
        Initialize the camera manager.
        
        Args:
            session_dir (str): Directory to save session data
        """
        self.session_dir = session_dir
        self.participants = {}
        self.frame_queue = Queue(maxsize=100)
        self.is_running = False
        
        logger.info("Initialized Camera Manager")
    
    def add_participant(self, participant_id, device_id):
        """
        Add a participant with associated camera.
        
        Args:
            participant_id (str): Identifier for the participant
            device_id (int): Camera device ID
        """
        if participant_id in self.participants:
            logger.warning(f"Participant {participant_id} already exists. Stopping existing camera.")
            self.participants[participant_id].stop()
        
        self.participants[participant_id] = ParticipantCamera(
            device_id=device_id,
            participant_id=participant_id,
            session_dir=self.session_dir,
            frame_queue=self.frame_queue
        )
        
        logger.info(f"Added participant {participant_id} with camera {device_id}")
        return True
    
    def start_all_cameras(self):
        """Start all participant cameras."""
        if not self.participants:
            logger.warning("No participants added. Cannot start cameras.")
            return False
        
        success = True
        for participant_id, camera in self.participants.items():
            try:
                camera.start()
            except Exception as e:
                log_exception(logger, e, f"Failed to start camera for {participant_id}")
                success = False
        
        self.is_running = success
        return success
    
    def stop_all_cameras(self):
        """Stop all participant cameras."""
        for participant_id, camera in self.participants.items():
            try:
                camera.stop()
            except Exception as e:
                log_exception(logger, e, f"Error stopping camera for {participant_id}")
        
        self.is_running = False
        logger.info("Stopped all cameras")
    
    def get_confusion_scores(self):
        """Get confusion scores for all participants."""
        scores = {}
        for participant_id, camera in self.participants.items():
            scores[participant_id] = camera.get_confusion_score()
        return scores
    
    def get_latest_frames(self):
        """Get the most recent frames from all cameras."""
        frames = {}
        for participant_id, camera in self.participants.items():
            frames[participant_id] = camera.get_current_frame()
        return frames
    
    def get_latest_features(self):
        """Get the most recent features from all cameras."""
        features = {}
        for participant_id, camera in self.participants.items():
            features[participant_id] = camera.get_latest_features()
        return features