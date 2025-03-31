"""
Camera capture module for the misalignment detection system.
Handles capturing frames from multiple cameras simultaneously.
"""
import time
import threading
import queue
import cv2
import numpy as np
from loguru import logger
from typing import Dict, List, Optional, Tuple, Union

from config import config


class Camera:
    """
    Class to handle operations for a single camera.
    """
    def __init__(self, camera_id: int, name: str = None):
        """
        Initialize a camera.
        
        Args:
            camera_id: The ID of the camera (usually 0, 1, 2, etc.)
            name: Optional name for the camera (e.g., "Person A", "Person B")
        """
        self.camera_id = camera_id
        self.name = name or f"Camera_{camera_id}"
        self.cap = None
        self.is_capturing = False
        self.capture_thread = None
        self.frame_queue = queue.Queue(maxsize=10)  # Limit queue size to prevent memory issues
        self.last_frame = None
        self.last_frame_time = 0
        self.frame_count = 0
        self.error_count = 0
        self.lock = threading.Lock()

    def start(self) -> bool:
        """
        Start capturing from the camera.
        
        Returns:
            bool: True if the camera was successfully started, False otherwise.
        """
        if self.is_capturing:
            logger.warning(f"{self.name}: Already capturing.")
            return True
            
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, config.camera.fps)
            
            if not self.cap.isOpened():
                logger.error(f"{self.name}: Failed to open camera.")
                return False
                
            # Start capture thread
            self.is_capturing = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            logger.info(f"{self.name}: Started capturing.")
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Error starting camera: {str(e)}")
            self.stop()
            return False
            
    def _capture_loop(self):
        """
        Main loop for capturing frames from the camera.
        Runs in a separate thread.
        """
        while self.is_capturing:
            try:
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    self.error_count += 1
                    if self.error_count > 5:
                        logger.warning(f"{self.name}: Multiple frame capture failures, attempting to restart...")
                        self._attempt_restart()
                    time.sleep(0.1)
                    continue
                    
                self.error_count = 0  # Reset error count on successful capture
                
                # Update frame info
                timestamp = time.time()
                with self.lock:
                    self.last_frame = frame.copy()
                    self.last_frame_time = timestamp
                    self.frame_count += 1
                
                # Add to queue, dropping oldest frame if full
                try:
                    self.frame_queue.put((timestamp, frame), block=False)
                except queue.Full:
                    try:
                        # Discard oldest frame
                        self.frame_queue.get(block=False)
                        self.frame_queue.put((timestamp, frame), block=False)
                    except (queue.Empty, queue.Full):
                        pass  # Rare race condition, just continue
                        
            except Exception as e:
                logger.error(f"{self.name}: Error in capture loop: {str(e)}")
                self.error_count += 1
                time.sleep(0.1)
                
            # Short sleep to prevent tight loop
            time.sleep(0.01)
            
    def _attempt_restart(self):
        """
        Attempt to restart the camera after failures.
        """
        logger.info(f"{self.name}: Attempting to restart camera...")
        try:
            if self.cap is not None:
                self.cap.release()
                
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                logger.error(f"{self.name}: Failed to reopen camera.")
            else:
                logger.info(f"{self.name}: Successfully reopened camera.")
                self.error_count = 0
        except Exception as e:
            logger.error(f"{self.name}: Error during camera restart: {str(e)}")
    
    def get_frame(self) -> Tuple[Optional[float], Optional[np.ndarray]]:
        """
        Get the most recent frame from the camera.
        
        Returns:
            tuple: (timestamp, frame) or (None, None) if no frame is available
        """
        with self.lock:
            if self.last_frame is None:
                return None, None
            return self.last_frame_time, self.last_frame.copy()
    
    def get_next_frame(self, timeout: float = 1.0) -> Tuple[Optional[float], Optional[np.ndarray]]:
        """
        Get the next frame from the queue.
        
        Args:
            timeout: Maximum time to wait for a frame (seconds)
            
        Returns:
            tuple: (timestamp, frame) or (None, None) if timeout occurs
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None, None
    
    def stop(self):
        """
        Stop the camera and release resources.
        """
        self.is_capturing = False
        
        # Wait for capture thread to end
        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
            
        # Release the camera
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        # Clear the queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get(block=False)
            except queue.Empty:
                break
                
        logger.info(f"{self.name}: Stopped capturing.")
        
    def is_working(self) -> bool:
        """
        Check if the camera is working properly.
        
        Returns:
            bool: True if the camera is working, False otherwise
        """
        # Camera is working if it's capturing and the error count is low
        if not self.is_capturing or self.cap is None:
            return False
            
        # Check if frames are being received
        with self.lock:
            if self.last_frame is None:
                return False
                
            # Check if the last frame is recent (within 5 seconds)
            if time.time() - self.last_frame_time > 5.0:
                return False
                
        return self.error_count <= 3


class CameraManager:
    """
    Manager for handling multiple cameras.
    """
    def __init__(self, camera_ids: List[int] = None, camera_names: List[str] = None):
        """
        Initialize the camera manager with multiple cameras.
        
        Args:
            camera_ids: List of camera IDs to use
            camera_names: Optional list of names for the cameras
        """
        if camera_ids is None:
            camera_ids = config.camera.camera_ids
            
        if camera_names is None:
            camera_names = [f"Person_{chr(65+i)}" for i in range(len(camera_ids))]
            
        if len(camera_names) < len(camera_ids):
            # Extend camera_names if needed
            camera_names.extend([f"Person_{chr(65+i)}" for i in range(len(camera_names), len(camera_ids))])
            
        self.cameras = {}
        for i, camera_id in enumerate(camera_ids):
            name = camera_names[i] if i < len(camera_names) else f"Camera_{camera_id}"
            self.cameras[name] = Camera(camera_id, name)
            
        self.timed_capture_thread = None
        self.is_timed_capturing = False
        self.capture_interval = config.camera.capture_interval
        self.synchronized_frame_queue = queue.Queue(maxsize=20)
        
    def start_all(self) -> Dict[str, bool]:
        """
        Start all cameras.
        
        Returns:
            dict: Map of camera names to success status
        """
        results = {}
        for name, camera in self.cameras.items():
            results[name] = camera.start()
        return results
        
    def stop_all(self):
        """
        Stop all cameras.
        """
        self.stop_timed_capture()
        for camera in self.cameras.values():
            camera.stop()
            
    def get_frames(self) -> Dict[str, Tuple[Optional[float], Optional[np.ndarray]]]:
        """
        Get the most recent frame from all cameras.
        
        Returns:
            dict: Map of camera names to (timestamp, frame) tuples
        """
        return {name: camera.get_frame() for name, camera in self.cameras.items()}
        
    def get_working_status(self) -> Dict[str, bool]:
        """
        Get the working status of all cameras.
        
        Returns:
            dict: Map of camera names to working status (True/False)
        """
        return {name: camera.is_working() for name, camera in self.cameras.items()}
        
    def start_timed_capture(self, interval: float = None) -> bool:
        """
        Start capturing frames at regular intervals.
        
        Args:
            interval: Capture interval in seconds (overrides config if provided)
            
        Returns:
            bool: True if timed capture was started, False otherwise
        """
        if self.is_timed_capturing:
            logger.warning("Timed capture already running.")
            return True
            
        # Set the capture interval
        if interval is not None:
            self.capture_interval = interval
        else:
            self.capture_interval = config.camera.capture_interval
            
        # Check if we have working cameras
        working_cameras = self.get_working_status()
        if not any(working_cameras.values()):
            logger.error("No working cameras available for timed capture.")
            return False
            
        # Start the timed capture thread
        self.is_timed_capturing = True
        self.timed_capture_thread = threading.Thread(target=self._timed_capture_loop, daemon=True)
        self.timed_capture_thread.start()
        logger.info(f"Started timed capture with interval {self.capture_interval} seconds.")
        return True
        
    def _timed_capture_loop(self):
        """
        Loop that captures frames from all cameras at regular intervals.
        """
        next_capture_time = time.time()
        
        while self.is_timed_capturing:
            current_time = time.time()
            
            # Check if it's time to capture
            if current_time >= next_capture_time:
                # Capture frames from all cameras
                frames = self.get_frames()
                
                # Add to synchronized frame queue
                try:
                    self.synchronized_frame_queue.put((current_time, frames), block=False)
                except queue.Full:
                    try:
                        # Discard oldest frame set
                        self.synchronized_frame_queue.get(block=False)
                        self.synchronized_frame_queue.put((current_time, frames), block=False)
                    except (queue.Empty, queue.Full):
                        pass
                        
                # Calculate next capture time
                next_capture_time = current_time + self.capture_interval
                
            # Short sleep to prevent tight loop
            time.sleep(min(0.1, max(0.001, next_capture_time - time.time())))
            
    def get_next_synchronized_frames(self, timeout: float = 1.0) -> Tuple[Optional[float], Optional[Dict[str, Tuple[float, np.ndarray]]]]:
        """
        Get the next set of synchronized frames from all cameras.
        
        Args:
            timeout: Maximum time to wait for frames (seconds)
            
        Returns:
            tuple: (timestamp, {camera_name: (timestamp, frame)}) or (None, None) if timeout occurs
        """
        try:
            return self.synchronized_frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None, None
            
    def stop_timed_capture(self):
        """
        Stop the timed capture.
        """
        self.is_timed_capturing = False
        
        # Wait for timed capture thread to end
        if self.timed_capture_thread is not None and self.timed_capture_thread.is_alive():
            self.timed_capture_thread.join(timeout=1.0)
            
        # Clear the queue
        while not self.synchronized_frame_queue.empty():
            try:
                self.synchronized_frame_queue.get(block=False)
            except queue.Empty:
                break
                
        logger.info("Stopped timed capture.")
        
    def set_capture_interval(self, interval: float):
        """
        Set the capture interval for timed capture.
        
        Args:
            interval: Capture interval in seconds
        """
        if interval < 0.1:
            logger.warning(f"Capture interval {interval} too small, setting to 0.1 seconds.")
            interval = 0.1
            
        self.capture_interval = interval
        logger.info(f"Set capture interval to {interval} seconds.")