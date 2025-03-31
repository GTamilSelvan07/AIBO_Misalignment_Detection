"""
Combined score processing for the misalignment detection system.
"""
import time
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from loguru import logger

from config import config
from src.scoring.camera_score import CameraScoreProcessor
from src.scoring.speech_score import SpeechScoreProcessor


class CombinedScoreProcessor:
    """
    Processes and manages combined misalignment scores.
    """
    def __init__(self, camera_processor: CameraScoreProcessor = None, 
                speech_processor: SpeechScoreProcessor = None):
        """
        Initialize the combined score processor.
        
        Args:
            camera_processor: Camera score processor
            speech_processor: Speech score processor
        """
        # Create processors if not provided
        self.camera_processor = camera_processor or CameraScoreProcessor()
        self.speech_processor = speech_processor or SpeechScoreProcessor()
        
        # Score history for each person
        self.score_history = {}  # {person_name: deque of (timestamp, score, camera_score, speech_score, details)}
        self.history_size = config.scoring.history_size
        
        # Weights for combining scores
        self.camera_weight = config.scoring.camera_weight
        self.speech_weight = config.scoring.speech_weight
        
        # Alert thresholds
        self.alert_threshold = config.scoring.alert_threshold
        
        # Recent alerts to avoid duplicates
        self.recent_alerts = {}  # {person_name: timestamp of last alert}
        self.alert_cooldown = 10.0  # seconds
        
        # LLM client for enriched analysis
        self.llm_client = None
        
    def set_llm_client(self, llm_client):
        """
        Set the LLM client for enriched analysis.
        
        Args:
            llm_client: LLM client instance
        """
        self.llm_client = llm_client
        
    def calculate_combined_score(self, camera_score: int, speech_score: int) -> int:
        """
        Calculate a combined score from camera and speech scores.
        
        Args:
            camera_score: Camera-based misalignment score (0-100)
            speech_score: Speech-based misalignment score (0-100)
            
        Returns:
            int: Combined score (0-100)
        """
        # Simple weighted average
        combined = (camera_score * self.camera_weight) + (speech_score * self.speech_weight)
        
        # Normalize to 0-100 range
        return int(min(100, max(0, combined)))
        
    def add_scores(self, person_name: str, 
                 camera_score: int, camera_details: Dict,
                 speech_score: int, speech_details: Dict, speech_text: str,
                 timestamp: float = None, do_llm_analysis: bool = False) -> Tuple[int, Dict]:
        """
        Add new scores for a person and calculate a combined score.
        
        Args:
            person_name: Identifier for the person
            camera_score: Camera-based misalignment score
            camera_details: Details of the camera score
            speech_score: Speech-based misalignment score
            speech_details: Details of the speech score
            speech_text: Transcribed text that was analyzed
            timestamp: Time of the scores (default: current time)
            do_llm_analysis: Whether to perform additional LLM analysis
            
        Returns:
            tuple: (combined_score, details)
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Add individual scores to their processors
        self.camera_processor.add_score(person_name, camera_score, camera_details, timestamp)
        self.speech_processor.add_score(person_name, speech_score, speech_details, speech_text, timestamp)
        
        # Calculate combined score
        combined_score = self.calculate_combined_score(camera_score, speech_score)
        
        # Initialize history for this person if needed
        if person_name not in self.score_history:
            self.score_history[person_name] = deque(maxlen=self.history_size)
            
        # Prepare details
        details = {
            "camera_score": camera_score,
            "camera_details": camera_details,
            "speech_score": speech_score,
            "speech_details": speech_details,
            "speech_text": speech_text,
            "weights": {
                "camera": self.camera_weight,
                "speech": self.speech_weight
            },
            "llm_analysis": None
        }
        
        # Perform LLM analysis if requested and LLM client is available
        if do_llm_analysis and self.llm_client is not None and combined_score >= 30:
            try:
                # Import here to avoid circular imports
                from src.llm.prompts import get_combined_analysis_prompt
                
                # Generate prompt
                prompt = get_combined_analysis_prompt(
                    camera_score, camera_details,
                    speech_score, speech_details,
                    speech_text
                )
                
                # Get LLM response
                response = self.llm_client.complete(prompt)
                
                # Parse response
                from src.llm.response_parser import parse_llm_response
                llm_result = parse_llm_response(response)
                
                if llm_result:
                    details["llm_analysis"] = llm_result
                    
                    # Adjust combined score based on LLM analysis
                    if "combined_score" in llm_result:
                        try:
                            llm_score = int(llm_result["combined_score"])
                            # Blend with our calculated score
                            combined_score = int(0.7 * combined_score + 0.3 * llm_score)
                        except (ValueError, TypeError):
                            pass
                    
            except Exception as e:
                logger.error(f"Error in LLM analysis for combined score: {str(e)}")
                
        # Add to history
        self.score_history[person_name].append(
            (timestamp, combined_score, camera_score, speech_score, details)
        )
        
        # Check for alerts
        self._check_for_alerts(person_name, combined_score, details, timestamp)
        
        return combined_score, details
        
    def _check_for_alerts(self, person_name: str, score: int, details: Dict, timestamp: float):
        """
        Check if a score should trigger an alert.
        
        Args:
            person_name: Identifier for the person
            score: Combined misalignment score
            details: Score details
            timestamp: Score timestamp
        """
        # Check if score is above threshold
        if score >= self.alert_threshold:
            # Check if we've alerted recently
            last_alert = self.recent_alerts.get(person_name, 0)
            if timestamp - last_alert > self.alert_cooldown:
                # Generate alert
                logger.warning(f"ALERT: High combined misalignment score for {person_name}: {score}")
                
                # Add speech text if available
                speech_text = details.get("speech_text")
                if speech_text:
                    logger.warning(f"Text: '{speech_text}'")
                    
                # Update last alert time
                self.recent_alerts[person_name] = timestamp
                
    def get_latest_score(self, person_name: str) -> Tuple[Optional[int], Optional[Dict], Optional[float]]:
        """
        Get the latest combined score for a person.
        
        Args:
            person_name: Identifier for the person
            
        Returns:
            tuple: (score, details, timestamp) or (None, None, None) if no scores
        """
        if person_name not in self.score_history or len(self.score_history[person_name]) == 0:
            return None, None, None
            
        timestamp, score, _, _, details = self.score_history[person_name][-1]
        return score, details, timestamp
        
    def get_smoothed_score(self, person_name: str, window_size: int = None) -> int:
        """
        Get a smoothed combined score over recent history.
        
        Args:
            person_name: Identifier for the person
            window_size: Number of recent scores to use (default: from config)
            
        Returns:
            int: Smoothed score (0-100)
        """
        if person_name not in self.score_history or len(self.score_history[person_name]) == 0:
            return 0
            
        if window_size is None:
            window_size = min(config.scoring.smoothing_window, len(self.score_history[person_name]))
        else:
            window_size = min(window_size, len(self.score_history[person_name]))
            
        # Get recent scores
        recent_scores = [s[1] for s in list(self.score_history[person_name])[-window_size:]]
        
        # Calculate weighted average with more recent scores weighted higher
        weights = [i+1 for i in range(len(recent_scores))]
        weighted_sum = sum(score * weight for score, weight in zip(recent_scores, weights))
        total_weight = sum(weights)
        
        return int(weighted_sum / total_weight)
        
    def get_history(self, person_name: str, max_items: int = None) -> List[Tuple[float, int, int, int, Dict]]:
        """
        Get the combined score history for a person.
        
        Args:
            person_name: Identifier for the person
            max_items: Maximum number of items to return
            
        Returns:
            list: List of (timestamp, combined_score, camera_score, speech_score, details) tuples
        """
        if person_name not in self.score_history:
            return []
            
        history = list(self.score_history[person_name])
        
        if max_items is not None and max_items < len(history):
            history = history[-max_items:]
            
        return history
        
    def clear_history(self, person_name: str = None):
        """
        Clear the score history for one or all persons.
        
        Args:
            person_name: Specific person to clear, or None for all
        """
        if person_name is None:
            # Clear all histories
            self.score_history = {}
            self.recent_alerts = {}
            self.camera_processor.clear_history()
            self.speech_processor.clear_history()
            logger.info("Cleared all score histories")
        elif person_name in self.score_history:
            # Clear specific person
            self.score_history[person_name] = deque(maxlen=self.history_size)
            if person_name in self.recent_alerts:
                del self.recent_alerts[person_name]
            self.camera_processor.clear_history(person_name)
            self.speech_processor.clear_history(person_name)
            logger.info(f"Cleared score history for {person_name}")
            
    def set_weights(self, camera_weight: float = None, speech_weight: float = None):
        """
        Set the weights for combining scores.
        
        Args:
            camera_weight: Weight for camera scores
            speech_weight: Weight for speech scores
        """
        if camera_weight is not None:
            self.camera_weight = max(0.0, min(1.0, camera_weight))
            
        if speech_weight is not None:
            self.speech_weight = max(0.0, min(1.0, speech_weight))
            
        # Normalize weights to sum to 1.0
        total = self.camera_weight + self.speech_weight
        if total > 0:
            self.camera_weight /= total
            self.speech_weight /= total
            
        logger.info(f"Set score weights: camera={self.camera_weight:.2f}, speech={self.speech_weight:.2f}")
        
    def set_alert_threshold(self, threshold: int):
        """
        Set the alert threshold.
        
        Args:
            threshold: New threshold (0-100)
        """
        self.alert_threshold = max(0, min(100, threshold))
        logger.info(f"Set combined score alert threshold to {self.alert_threshold}")
        
    def get_all_latest_scores(self) -> Dict[str, Tuple[int, Dict, float]]:
        """
        Get the latest combined scores for all persons.
        
        Returns:
            dict: {person_name: (score, details, timestamp)}
        """
        result = {}
        # Get all persons from both processors
        person_names = set(list(self.camera_processor.score_history.keys()) + 
                          list(self.speech_processor.score_history.keys()))
                          
        for person_name in person_names:
            score, details, timestamp = self.get_latest_score(person_name)
            if score is not None:
                result[person_name] = (score, details, timestamp)
                
        return result
        
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