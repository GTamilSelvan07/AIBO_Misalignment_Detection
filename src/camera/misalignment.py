"""
Visual misalignment detection using facial expressions.
Analyzes OpenFace features to identify confusion and misalignment.
"""
import time
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
from loguru import logger

from config import config


class MisalignmentDetector:
    """
    Detects misalignment and confusion based on facial expressions.
    Uses Action Units (AUs) from OpenFace to identify signs of confusion.
    """
    def __init__(self):
        """
        Initialize the misalignment detector.
        """
        # Load confusion-related AUs and their weights from config
        self.confusion_aus = config.camera.confusion_aus
        self.au_weights = config.camera.au_weights
        
        # History of recent scores for smoothing and personalization
        self.person_history = {}  # {person_name: deque of scores}
        self.baseline_au_values = {}  # {person_name: {AU: baseline_value}}
        self.history_window = config.scoring.smoothing_window
        self.baseline_period = config.scoring.baseline_period
        self.adaptation_rate = config.scoring.adaptation_rate
        
        # Timestamps for baseline computation
        self.baseline_start_times = {}  # {person_name: start_time}
        
    def _initialize_person(self, person_name: str):
        """
        Initialize tracking for a new person.
        
        Args:
            person_name: Identifier for the person
        """
        if person_name not in self.person_history:
            self.person_history[person_name] = deque(maxlen=self.history_window)
            self.baseline_au_values[person_name] = {}
            self.baseline_start_times[person_name] = time.time()
            logger.info(f"Initialized tracking for {person_name}")
            
    def update_baseline(self, person_name: str, aus: Dict[int, float]):
        """
        Update the baseline AU values for a person.
        
        Args:
            person_name: Identifier for the person
            aus: Dictionary of {AU_number: intensity}
        """
        self._initialize_person(person_name)
        
        # Skip if AUs are empty
        if not aus:
            return
            
        baseline_time = self.baseline_start_times[person_name]
        current_time = time.time()
        
        # If still in baseline period, accumulate values
        if current_time - baseline_time < self.baseline_period:
            for au_num, intensity in aus.items():
                if au_num in self.baseline_au_values[person_name]:
                    # Running average
                    old_value = self.baseline_au_values[person_name][au_num]
                    count = self.baseline_au_values.get(f"{au_num}_count", 1)
                    new_value = (old_value * count + intensity) / (count + 1)
                    self.baseline_au_values[person_name][au_num] = new_value
                    self.baseline_au_values[person_name][f"{au_num}_count"] = count + 1
                else:
                    # First value
                    self.baseline_au_values[person_name][au_num] = intensity
                    self.baseline_au_values[person_name][f"{au_num}_count"] = 1
                    
            logger.debug(f"Updated baseline for {person_name}, elapsed: {current_time - baseline_time:.1f}s")
        else:
            # After baseline period, gradually adapt to new observations
            if current_time - baseline_time < self.baseline_period + 10:  # Log once when baseline complete
                logger.info(f"Baseline collection completed for {person_name}")
                
            # Gradual adaptation
            for au_num, intensity in aus.items():
                if au_num in self.baseline_au_values[person_name]:
                    old_value = self.baseline_au_values[person_name][au_num]
                    self.baseline_au_values[person_name][au_num] = (
                        (1 - self.adaptation_rate) * old_value + 
                        self.adaptation_rate * intensity
                    )
                else:
                    self.baseline_au_values[person_name][au_num] = intensity
                    
    def compute_misalignment_score(self, person_name: str, face_data: Optional[Dict]) -> Tuple[int, Dict]:
        """
        Compute a misalignment score (0-100) based on facial expressions.
        
        Args:
            person_name: Identifier for the person
            face_data: Face features from face detector
            
        Returns:
            tuple: (misalignment_score, details)
        """
        self._initialize_person(person_name)
        
        # Return zero score if no face data
        if face_data is None:
            return 0, {
                "score": 0,
                "active_aus": {},
                "reason": "No face detected"
            }
            
        # Extract AUs from face data
        aus = {}
        for i in self.confusion_aus:
            au_key = f'AU{i}'
            if au_key in face_data:
                aus[i] = float(face_data[au_key])
                
        # If no AUs were detected, return zero score
        if not aus:
            return 0, {
                "score": 0,
                "active_aus": {},
                "reason": "No Action Units detected"
            }
            
        # Update baseline values
        self.update_baseline(person_name, aus)
        
        # Calculate misalignment score
        score = 0
        active_aus = {}
        max_possible_score = 0
        
        for au_num in self.confusion_aus:
            if au_num in aus:
                # Get the intensity of this AU
                intensity = aus[au_num]
                
                # Get baseline for this AU
                baseline = self.baseline_au_values[person_name].get(au_num, 0)
                
                # Calculate deviation from baseline
                deviation = max(0, intensity - baseline)
                
                # Get weight for this AU
                weight = self.au_weights.get(au_num, 1.0)
                
                # Add to score
                au_score = deviation * weight * 100
                score += au_score
                max_possible_score += weight * 100
                
                # Track active AUs (those that contribute to confusion score)
                if deviation > 0.1:  # Only consider significant deviations
                    active_aus[au_num] = {
                        "intensity": intensity,
                        "baseline": baseline,
                        "deviation": deviation,
                        "contribution": au_score
                    }
                    
        # Normalize score to 0-100 range
        if max_possible_score > 0:
            normalized_score = min(100, int((score / max_possible_score) * 100))
        else:
            normalized_score = 0
            
        # Add to history for smoothing
        self.person_history[person_name].append(normalized_score)
        
        # Calculate smoothed score
        if len(self.person_history[person_name]) > 0:
            smoothed_score = int(np.mean(self.person_history[person_name]))
        else:
            smoothed_score = normalized_score
            
        # Prepare result details
        result = {
            "raw_score": normalized_score,
            "smoothed_score": smoothed_score,
            "active_aus": active_aus,
            "au_weights": {k: v for k, v in self.au_weights.items() if k in self.confusion_aus},
            "person_name": person_name,
            "baseline_status": "established" if time.time() - self.baseline_start_times[person_name] > self.baseline_period else "collecting"
        }
        
        return smoothed_score, result
        
    def analyze_multiple_faces(self, face_results: Dict[str, Tuple[bool, Optional[Dict]]]) -> Dict[str, Tuple[int, Dict]]:
        """
        Analyze multiple faces for misalignment.
        
        Args:
            face_results: Results from FaceDetector.detect_faces_multi
            
        Returns:
            dict: {person_name: (misalignment_score, details)}
        """
        results = {}
        for person_name, (success, face_data) in face_results.items():
            if success:
                results[person_name] = self.compute_misalignment_score(person_name, face_data)
            else:
                results[person_name] = self.compute_misalignment_score(person_name, None)
                
        return results
        
    def get_misalignment_description(self, score: int) -> str:
        """
        Get a textual description of the misalignment score.
        
        Args:
            score: Misalignment score (0-100)
            
        Returns:
            str: Description of misalignment level
        """
        if score < 20:
            return "Low misalignment"
        elif score < 50:
            return "Moderate misalignment"
        elif score < 80:
            return "High misalignment"
        else:
            return "Severe misalignment"
            
    def reset_baseline(self, person_name: str = None):
        """
        Reset the baseline for one or all persons.
        
        Args:
            person_name: Specific person to reset, or None for all
        """
        if person_name is None:
            # Reset all
            self.baseline_au_values = {}
            self.baseline_start_times = {}
            for name in self.person_history:
                self.baseline_start_times[name] = time.time()
                self.baseline_au_values[name] = {}
            logger.info("Reset baselines for all persons")
        elif person_name in self.person_history:
            # Reset specific person
            self.baseline_au_values[person_name] = {}
            self.baseline_start_times[person_name] = time.time()
            logger.info(f"Reset baseline for {person_name}")