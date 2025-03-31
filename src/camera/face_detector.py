"""
Face detection and OpenFace feature extraction for the misalignment detection system.
"""
import os
import time
import threading
import numpy as np
import cv2
from loguru import logger
from typing import Dict, List, Optional, Tuple, Union
from config import config

# Try to import py_feat, but provide a fallback if not available
PY_FEAT_AVAILABLE = False
try:
    from py_feat import Detector
    PY_FEAT_AVAILABLE = True
except ImportError:
    logger.warning("py_feat package not found. Using fallback face detection with OpenCV only.")
    logger.warning("To install py_feat, run: pip install py-feat")


class FaceDetector:
    """
    Class for detecting faces and extracting facial features using OpenFace (via py-feat).
    Falls back to basic OpenCV face detection if py-feat is not available.
    """
    def __init__(self, detection_threshold: float = None):
        """
        Initialize the face detector.
        
        Args:
            detection_threshold: Confidence threshold for face detection
        """
        self.detection_threshold = detection_threshold or config.camera.detection_threshold
        self.detector = None
        self.lock = threading.Lock()
        self.is_initialized = False
        self.initialization_thread = None
        self.initialization_error = None
        self.using_fallback = not PY_FEAT_AVAILABLE
        
        # Load OpenCV's face detection for fallback
        self.cv_face_cascade = None
        if self.using_fallback:
            try:
                # Load OpenCV's built-in face detector as fallback
                model_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                if os.path.exists(model_path):
                    self.cv_face_cascade = cv2.CascadeClassifier(model_path)
                    logger.info("Initialized OpenCV fallback face detector")
                    self.is_initialized = True
                else:
                    logger.error(f"OpenCV face model not found at: {model_path}")
                    self.initialization_error = f"OpenCV face model not found"
            except Exception as e:
                logger.error(f"Error initializing OpenCV fallback face detector: {str(e)}")
                self.initialization_error = f"Error initializing OpenCV fallback: {str(e)}"
        else:
            # Start initialization in a separate thread to avoid blocking
            self.initialization_thread = threading.Thread(target=self._initialize_detector, daemon=True)
            self.initialization_thread.start()
        
    def _initialize_detector(self):
        """
        Initialize the Py-Feat detector with OpenFace backend.
        This can take a few seconds, so it's done in a separate thread.
        """
        if self.using_fallback:
            return
            
        try:
            logger.info("Initializing face detector with OpenFace backend...")
            start_time = time.time()
            
            # Initialize the detector with OpenFace backend
            # Note: py-feat uses a variety of backends, but we're specifically using OpenFace
            self.detector = Detector(
                face_model="retinaface",  # Fast face detection
                landmark_model="mobilefacenet",  # Fast landmark detection
                au_model="jaanet",  # Action Units model (OpenFace compatible)
                emotion_model="resmasknet",  # Emotion detection
                facepose_model="img2pose",  # Head pose estimation
            )
            
            logger.info(f"Face detector initialization completed in {time.time() - start_time:.2f} seconds")
            self.is_initialized = True
            
        except Exception as e:
            error_msg = f"Error initializing face detector: {str(e)}"
            logger.error(error_msg)
            self.initialization_error = error_msg
            self.using_fallback = True
            
            # Try to initialize OpenCV fallback
            try:
                model_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                if os.path.exists(model_path):
                    self.cv_face_cascade = cv2.CascadeClassifier(model_path)
                    logger.info("Initialized OpenCV fallback face detector")
                    self.is_initialized = True
                else:
                    logger.error(f"OpenCV face model not found at: {model_path}")
            except Exception as cv_e:
                logger.error(f"Error initializing OpenCV fallback: {str(cv_e)}")
            
    def wait_for_initialization(self, timeout: float = 30.0) -> bool:
        """
        Wait for the detector to initialize.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        if self.is_initialized:
            return True
            
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_initialized:
                return True
            if self.initialization_error:
                return False
            time.sleep(0.1)
            
        # Timeout occurred
        logger.error(f"Face detector initialization timed out after {timeout} seconds")
        return False
        
    def detect_face(self, frame: np.ndarray) -> Tuple[bool, Optional[Dict]]:
        """
        Detect a face in the frame and extract features.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            tuple: (success, features) where features include face location, landmarks, AUs, etc.
        """
        if not self.is_initialized:
            if not self.wait_for_initialization(timeout=10.0):
                return False, None
                
        if frame is None:
            logger.warning("Cannot detect face: frame is None")
            return False, None
            
        try:
            # Use OpenCV fallback if py_feat is not available
            if self.using_fallback:
                return self._detect_face_opencv(frame)
            else:
                return self._detect_face_pyfeat(frame)
                
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            return False, None
            
    def _detect_face_pyfeat(self, frame: np.ndarray) -> Tuple[bool, Optional[Dict]]:
        """
        Detect a face using py_feat.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            tuple: (success, features)
        """
        # Convert to RGB for py-feat
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        with self.lock:
            # Detect faces and extract features
            result = self.detector.detect_image(rgb_frame)
            
        # Check if a face was detected
        if result.empty:
            return False, None
            
        # Extract the face with highest confidence
        max_confidence_idx = result['FaceRectConfidence'].idxmax()
        face_data = result.loc[max_confidence_idx].to_dict()
        
        # Only proceed if confidence is above threshold
        if face_data['FaceRectConfidence'] < self.detection_threshold:
            return False, None
            
        return True, face_data
        
    def _detect_face_opencv(self, frame: np.ndarray) -> Tuple[bool, Optional[Dict]]:
        """
        Detect a face using OpenCV as fallback.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            tuple: (success, features)
        """
        if self.cv_face_cascade is None:
            return False, None
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.cv_face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return False, None
            
        # Get the largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        
        # Create a simplified face data dict with just the rectangle
        face_data = {
            'FaceRectX': float(x),
            'FaceRectY': float(y),
            'FaceRectWidth': float(w),
            'FaceRectHeight': float(h),
            'FaceRectConfidence': 1.0,  # Placeholder confidence
            'using_fallback': True
        }
        
        # Add some placeholder AUs with zero intensity
        for au_num in range(1, 28):  # OpenFace typically detects AUs 1-27
            face_data[f'AU{au_num}'] = 0.0
            
        return True, face_data
        
    def detect_faces_multi(self, frames: Dict[str, np.ndarray]) -> Dict[str, Tuple[bool, Optional[Dict]]]:
        """
        Detect faces in multiple frames.
        
        Args:
            frames: Dictionary of {camera_name: frame}
            
        Returns:
            dict: Dictionary of {camera_name: (success, features)}
        """
        results = {}
        for name, frame in frames.items():
            results[name] = self.detect_face(frame)
        return results
        
    def draw_face_landmarks(self, frame: np.ndarray, face_data: Dict) -> np.ndarray:
        """
        Draw face landmarks and AU information on the frame.
        
        Args:
            frame: Input frame (BGR format)
            face_data: Face features from detect_face
            
        Returns:
            np.ndarray: Frame with landmarks and AUs drawn
        """
        if frame is None or face_data is None:
            return frame
            
        try:
            # Create a copy of the frame
            display_frame = frame.copy()
            
            # Extract face rectangle
            x = int(face_data['FaceRectX'])
            y = int(face_data['FaceRectY'])
            w = int(face_data['FaceRectWidth'])
            h = int(face_data['FaceRectHeight'])
            
            # Draw face rectangle
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Check if using fallback
            if face_data.get('using_fallback', False):
                # In fallback mode, we don't have landmarks or AUs
                cv2.putText(display_frame, "Basic Detection (Fallback)", 
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                return display_frame
                
            # Draw landmarks if available
            if 'x_0' in face_data:
                num_landmarks = 68  # Number of landmarks in OpenFace
                for i in range(num_landmarks):
                    x_key = f'x_{i}'
                    y_key = f'y_{i}'
                    if x_key in face_data and y_key in face_data:
                        lx = int(face_data[x_key])
                        ly = int(face_data[y_key])
                        cv2.circle(display_frame, (lx, ly), 1, (0, 0, 255), -1)
                        
            # Draw Action Units (AUs) information for confusion indicators
            au_names = {
                4: "Brow Lowerer",
                7: "Lid Tightener",
                9: "Nose Wrinkler",
                14: "Dimpler",
                15: "Lip Corner Depressor",
                17: "Chin Raiser",
                23: "Lip Tightener",
                24: "Lip Pressor"
            }
            
            # List to collect active AUs
            active_aus = []
            
            # Check for active AUs
            for au_num, au_name in au_names.items():
                au_key = f'AU{au_num}'
                if au_key in face_data and face_data[au_key] > 0.3:  # Threshold for considering an AU active
                    intensity = face_data[au_key]
                    active_aus.append((au_num, au_name, intensity))
            
            # Draw active AUs on the frame
            y_offset = 30
            for au_num, au_name, intensity in active_aus:
                text = f"AU{au_num} ({au_name}): {intensity:.2f}"
                cv2.putText(display_frame, text, (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                y_offset += 20
                
            # Draw head pose if available
            if all(k in face_data for k in ['pitch', 'yaw', 'roll']):
                pose_text = f"Pose: P:{face_data['pitch']:.1f} Y:{face_data['yaw']:.1f} R:{face_data['roll']:.1f}"
                cv2.putText(display_frame, pose_text, (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
            return display_frame
            
        except Exception as e:
            logger.error(f"Error drawing face landmarks: {str(e)}")
            return frame
            
    def extract_aus(self, face_data: Dict) -> Dict[int, float]:
        """
        Extract Action Units (AUs) from face data.
        
        Args:
            face_data: Face features from detect_face
            
        Returns:
            dict: Dictionary of {AU_number: intensity}
        """
        if face_data is None:
            return {}
            
        aus = {}
        for i in range(1, 28):  # OpenFace can detect AUs 1-27
            au_key = f'AU{i}'
            if au_key in face_data:
                aus[i] = float(face_data[au_key])
                
        return aus