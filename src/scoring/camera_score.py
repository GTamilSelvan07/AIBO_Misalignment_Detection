"""
Camera score processing for the misalignment detection system.
"""
import time
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from loguru import logger

from config import config


class CameraScoreProcessor:
    """
    Processes and manages camera-based misalignment scores.
    """
    def __init__(self):
        """
        Initialize the camera score processor.
        """
        # Score history for each person
        self.score_history = {}  # {person_name: deque of (timestamp, score, details)}
        self.history_size = config.scoring.history_size
        
        # Alert thresholds
        self.alert_threshold = config.scoring.alert_threshold
        
        # Recent alerts to avoid duplicates
        self.recent_alerts = {}  # {person_name: timestamp of last alert}
        self.alert_cooldown = 10.0  # seconds
        
    def add_score(self, person_name: str, score: int, details: Dict, timestamp: float = None):
        """
        Add a new camera score for a person.
        
        Args:
            person_name: Identifier for the person
            score: Misalignment score (0-100)
            details: Details of the score
            timestamp: Time of the score (default: current time)
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Initialize history for this person if needed
        if person_name not in self.score_history:
            self.score_history[person_name] = deque(maxlen=self.history_size)
            
        # Add to history
        self.score_history[person_name].append((timestamp, score, details))
        
        # Check for alerts
        self._check_for_alerts(person_name, score, details, timestamp)
        
    def _check_for_alerts(self, person_name: str, score: int, details: Dict, timestamp: float):
        """
        Check if a score should trigger an alert.
        
        Args:
            person_name: Identifier for the person
            score: Misalignment score
            details: Score details
            timestamp: Score timestamp
        """
        # Check if score is above threshold
        if score >= self.alert_threshold:
            # Check if we've alerted recently
            last_alert = self.recent_alerts.get(person_name, 0)
            if timestamp - last_alert > self.alert_cooldown:
                # Generate alert
                logger.warning(f"ALERT: High camera misalignment score for {person_name}: {score}")
                
                # Update last alert time
                self.recent_alerts[person_name] = timestamp
                
    def get_latest_score(self, person_name: str) -> Tuple[Optional[int], Optional[Dict], Optional[float]]:
        """
        Get the latest score for a person.
        
        Args:
            person_name: Identifier for the person
            
        Returns:
            tuple: (score, details, timestamp) or (None, None, None) if no scores
        """
        if person_name not in self.score_history or len(self.score_history[person_name]) == 0:
            return None, None, None
            
        timestamp, score, details = self.score_history[person_name][-1]
        return score, details, timestamp
        
    def get_smoothed_score(self, person_name: str, window_size: int = None) -> int:
        """
        Get a smoothed score over recent history.
        
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
        
    def get_score_trend(self, person_name: str, window_size: int = 5) -> str:
        """
        Get the trend of scores for a person.
        
        Args:
            person_name: Identifier for the person
            window_size: Number of recent scores to consider
            
        Returns:
            str: 'increasing', 'decreasing', 'stable', or 'unknown'
        """
        if person_name not in self.score_history or len(self.score_history[person_name]) < 2:
            return 'unknown'
            
        # Get recent scores
        history = list(self.score_history[person_name])
        if len(history) < window_size:
            window_size = len(history)
            
        recent_scores = [s[1] for s in history[-window_size:]]
        
        # Compute trend
        if len(recent_scores) < 2:
            return 'unknown'
            
        # Simple trend calculation using first and last scores
        first = recent_scores[0]
        last = recent_scores[-1]
        
        if abs(last - first) < 10:
            return 'stable'
        elif last > first:
            return 'increasing'
        else:
            return 'decreasing'
            
    def get_history(self, person_name: str, max_items: int = None) -> List[Tuple[float, int, Dict]]:
        """
        Get the score history for a person.
        
        Args:
            person_name: Identifier for the person
            max_items: Maximum number of items to return
            
        Returns:
            list: List of (timestamp, score, details) tuples
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
            logger.info("Cleared all camera score histories")
        elif person_name in self.score_history:
            # Clear specific person
            self.score_history[person_name] = deque(maxlen=self.history_size)
            if person_name in self.recent_alerts:
                del self.recent_alerts[person_name]
            logger.info(f"Cleared camera score history for {person_name}")
            
    def set_alert_threshold(self, threshold: int):
        """
        Set the alert threshold.
        
        Args:
            threshold: New threshold (0-100)
        """
        self.alert_threshold = max(0, min(100, threshold))
        logger.info(f"Set camera score alert threshold to {self.alert_threshold}")
        
    def get_all_latest_scores(self) -> Dict[str, Tuple[int, Dict, float]]:
        """
        Get the latest scores for all persons.
        
        Returns:
            dict: {person_name: (score, details, timestamp)}
        """
        result = {}
        for person_name in self.score_history:
            score, details, timestamp = self.get_latest_score(person_name)
            if score is not None:
                result[person_name] = (score, details, timestamp)
        return result