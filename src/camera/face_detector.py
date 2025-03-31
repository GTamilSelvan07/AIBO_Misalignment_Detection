"""
Face detection and feature extraction for the misalignment detection system.
Provides both OpenFace integration (via py-feat) and a fallback using OpenCV.
"""
import os
import time
import threading
import numpy as np
import cv2
from loguru import logger
from typing import Dict, List, Optional, Tuple, Union
from config import config, MODELS_DIR

# Try to import py-feat, but use fallback if not available
PY_FEAT_AVAILABLE = False
try:
    from feat import Detector
    PY_FEAT_AVAILABLE = True
except ImportError:
    logger.warning("py_feat package not found. Using fallback face detection with OpenCV only.")
    logger.warning("To install py_feat, run: pip install py-feat")


class FaceDetector:
    """
    Class for detecting faces and extracting facial features.
    Uses py-feat's OpenFace integration if available, otherwise falls back to OpenCV.
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
        self.use_py_feat = PY_FEAT_AVAILABLE
        
        # For OpenCV fallback
        self.face_cascade = None
        self.landmark_predictor = None
        
        # Start initialization in a separate thread to avoid blocking
        self.initialization_thread = threading.Thread(target=self._initialize_detector, daemon=True)
        self.initialization_thread.start()
        
    def _initialize_detector(self):
        """
        Initialize the detector based on available libraries.
        """
        try:
            if self.use_py_feat:
                logger.info("Initializing face detector with OpenFace backend via py-feat...")
                start_time = time.time()
                
                # Initialize the detector with OpenFace backend
                self.detector = Detector(
                    face_model="retinaface",  # Fast face detection
                    landmark_model="mobilefacenet",  # Fast landmark detection
                    au_model="jaanet",  # Action Units model (OpenFace compatible)
                    emotion_model="resmasknet",  # Emotion detection
                    facepose_model="img2pose",  # Head pose estimation
                )
                
                logger.info(f"Face detector initialization completed in {time.time() - start_time:.2f} seconds")
            else:
                # Fallback to OpenCV
                logger.info("Initializing OpenCV fallback face detector...")
                
                # Load Haar cascades for face detection
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                
                # Try to load dlib's facial landmark predictor if available
                try:
                    import dlib
                    # Check for the shape predictor file
                    shape_predictor_path = MODELS_DIR / "shape_predictor_68_face_landmarks.dat"
                    
                    # Download the shape predictor if it doesn't exist
                    if not os.path.exists(shape_predictor_path):
                        logger.warning(f"Facial landmark predictor not found at {shape_predictor_path}")
                        logger.warning("Facial landmarks will not be available.")
                    else:
                        self.landmark_predictor = dlib.shape_predictor(str(shape_predictor_path))
                        logger.info("Loaded dlib facial landmark predictor")
                        
                except ImportError:
                    logger.warning("dlib package not found. Facial landmarks will not be available.")
                    logger.warning("To enable landmarks, install dlib: pip install dlib")
                
                logger.info("Initialized OpenCV fallback face detector")
                
            self.is_initialized = True
            
        except Exception as e:
            error_msg = f"Error initializing face detector: {str(e)}"
            logger.error(error_msg)
            self.initialization_error = error_msg
            
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
        
    def detect_face_opencv(self, frame: np.ndarray) -> Tuple[bool, Optional[Dict]]:
        """
        Detect a face using OpenCV's Haar cascade classifier.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            tuple: (success, features) where features include face location, landmarks, etc.
        """
        if frame is None:
            return False, None
            
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) == 0:
                return False, None
                
            # Get the largest face
            if len(faces) > 1:
                # Sort by area (width * height)
                faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
                
            # Extract face coordinates
            x, y, w, h = faces[0]
            
            # Create face data dictionary
            face_data = {
                'FaceRectX': int(x),
                'FaceRectY': int(y),
                'FaceRectWidth': int(w),
                'FaceRectHeight': int(h),
                'FaceRectConfidence': 0.9,  # Hardcoded confidence for Haar cascade
            }
            
            # Add facial landmarks if available
            if self.landmark_predictor is not None:
                try:
                    import dlib
                    # Convert to dlib rectangle
                    rect = dlib.rectangle(x, y, x + w, y + h)
                    
                    # Get landmarks
                    landmarks = self.landmark_predictor(gray, rect)
                    
                    # Add landmarks to face data
                    for i in range(68):
                        face_data[f'x_{i}'] = landmarks.part(i).x
                        face_data[f'y_{i}'] = landmarks.part(i).y
                        
                except Exception as e:
                    logger.warning(f"Error detecting landmarks: {str(e)}")
                    
            # Simulate Action Units for facial expressions
            # This is a very basic simulation based on facial geometry
            # In a real system, these would come from OpenFace/py-feat
            self._add_simulated_action_units(face_data, frame, gray, x, y, w, h)
            
            return True, face_data
            
        except Exception as e:
            logger.error(f"Error in OpenCV face detection: {str(e)}")
            return False, None
            
    def _add_simulated_action_units(self, face_data: Dict, frame: np.ndarray, 
                                   gray: np.ndarray, x: int, y: int, w: int, h: int):
        """
        Add simulated Action Units (AUs) to the face data.
        
        This is a very basic approximation based on image features.
        Real AU detection requires specialized models like OpenFace.
        
        Args:
            face_data: Face data dictionary to update
            frame: Original frame
            gray: Grayscale frame
            x, y, w, h: Face rectangle coordinates
        """
        # Create simulated AU values
        face_roi = gray[y:y+h, x:x+w]
        
        # Basic AU simulation 
        # These are very rough approximations and not accurate compared to real AUs
        
        # AU4 (Brow Lowerer) - approximated by looking at gradients in upper face
        try:
            upper_face = face_roi[0:int(h/3), :]
            upper_grad = cv2.Sobel(upper_face, cv2.CV_64F, 0, 1, ksize=3)
            au4_value = min(1.0, np.mean(np.abs(upper_grad)) / 20.0)
            face_data['AU4'] = float(au4_value)
        except:
            face_data['AU4'] = 0.0
            
        # AU7 (Lid Tightener) - approximated by edge detection around eyes
        try:
            eye_region = face_roi[int(h/4):int(h/2), :]
            eye_edges = cv2.Canny(eye_region, 100, 200)
            au7_value = min(1.0, np.sum(eye_edges > 0) / (eye_region.size * 0.1))
            face_data['AU7'] = float(au7_value)
        except:
            face_data['AU7'] = 0.0
            
        # AU9 (Nose Wrinkler) - approximated by texture in nose region
        try:
            nose_region = face_roi[int(h/3):int(2*h/3), int(w/3):int(2*w/3)]
            nose_laplacian = cv2.Laplacian(nose_region, cv2.CV_64F)
            au9_value = min(1.0, np.var(nose_laplacian) / 100.0)
            face_data['AU9'] = float(au9_value)
        except:
            face_data['AU9'] = 0.0
            
        # Add other simulated AUs used for confusion detection
        aus = config.camera.confusion_aus
        
        # Random low values for other AUs
        for au in aus:
            if f'AU{au}' not in face_data:
                # Use random low values for simplicity
                # In a real system, these would be computed by OpenFace
                random_val = np.random.uniform(0.0, 0.3)
                face_data[f'AU{au}'] = float(random_val)
                
        # Add an intensity boost to AU4, AU7 when head is tilted
        # (a rough proxy for confusion)
        try:
            if 'x_0' in face_data and 'y_0' in face_data:
                # Check head tilt using facial landmarks
                left_eye = np.array([face_data['x_36'], face_data['y_36']])
                right_eye = np.array([face_data['x_45'], face_data['y_45']])
                
                eye_angle = np.abs(np.arctan2(
                    right_eye[1] - left_eye[1],
                    right_eye[0] - left_eye[0]
                ) * 180 / np.pi)
                
                # Boost confusion AUs if head is tilted
                if eye_angle > 5:
                    tilt_factor = min(1.0, eye_angle / 15.0)
                    face_data['AU4'] = min(1.0, face_data['AU4'] + tilt_factor * 0.3)
                    face_data['AU7'] = min(1.0, face_data['AU7'] + tilt_factor * 0.3)
        except:
            pass
            
    def detect_face(self, frame: np.ndarray) -> Tuple[bool, Optional[Dict]]:
        """
        Detect a face in the frame and extract facial features.
        
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
            # Use appropriate detection method based on available libraries
            if self.use_py_feat:
                # Use py-feat
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
            else:
                # Use OpenCV fallback
                return self.detect_face_opencv(frame)
                
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            return False, None
            
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
            x = int(face_data.get('FaceRectX', 0))
            y = int(face_data.get('FaceRectY', 0))
            w = int(face_data.get('FaceRectWidth', 0))
            h = int(face_data.get('FaceRectHeight', 0))
            
            # Draw face rectangle
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
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