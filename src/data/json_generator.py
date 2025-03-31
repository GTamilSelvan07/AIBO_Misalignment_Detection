"""
JSON generator for API communication in the misalignment detection system.
"""
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger


class JsonGenerator:
    """
    Generates standardized JSON output for the misalignment detection system.
    """
    def __init__(self, session_id: str = None):
        """
        Initialize the JSON generator.
        
        Args:
            session_id: Session identifier (generated if None)
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.session_start_time = time.time()
        
    def generate_misalignment_json(self, 
                                  person_name: str,
                                  camera_score: int,
                                  camera_details: Dict,
                                  speech_score: int,
                                  speech_details: Dict,
                                  transcript: str,
                                  combined_score: int, 
                                  timestamp: float = None) -> Dict:
        """
        Generate a standardized JSON for misalignment detection results.
        
        Args:
            person_name: Identifier for the person
            camera_score: Camera-based misalignment score
            camera_details: Details of the camera score
            speech_score: Speech-based misalignment score
            speech_details: Details of the speech score
            transcript: Transcribed text
            combined_score: Combined misalignment score
            timestamp: Time of the detection (default: current time)
            
        Returns:
            dict: Formatted JSON object
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Format timestamp as ISO 8601
        iso_timestamp = datetime.fromtimestamp(timestamp).isoformat() + "Z"
        
        # Extract detected issues
        detected_issues = []
        
        # Add camera-based issues
        if camera_details and "active_aus" in camera_details:
            active_aus = camera_details["active_aus"]
            if active_aus:
                # Get top 3 most significant AUs
                top_aus = sorted(
                    active_aus.items(), 
                    key=lambda x: x[1].get("contribution", 0),
                    reverse=True
                )[:3]
                
                # Create a camera-based issue
                visual_cues = [f"AU{au}" for au, _ in top_aus]
                
                detected_issues.append({
                    "time": datetime.fromtimestamp(timestamp).strftime("%H:%M:%S"),
                    "text_segment": "",
                    "misalignment_type": "visual confusion",
                    "confidence": camera_score / 100.0,
                    "visual_cues": visual_cues
                })
                
        # Add speech-based issues
        if speech_details:
            llm_analysis = speech_details.get("llm_analysis", {})
            indicators = []
            
            if llm_analysis and "indicators" in llm_analysis:
                indicators = llm_analysis.get("indicators", [])
                
            if indicators and transcript:
                detected_issues.append({
                    "time": datetime.fromtimestamp(timestamp).strftime("%H:%M:%S"),
                    "text_segment": transcript,
                    "misalignment_type": llm_analysis.get("misalignment_type", "verbal confusion"),
                    "confidence": llm_analysis.get("confidence", speech_score / 100.0),
                    "indicators": indicators
                })
                
        # Construct the JSON object
        result = {
            "timestamp": iso_timestamp,
            "session_id": self.session_id,
            "person_id": person_name,
            "camera_score": camera_score,
            "speech_score": speech_score,
            "combined_score": combined_score,
            "misalignment_context": {
                "transcript": transcript,
                "detected_issues": detected_issues
            },
            "technical_metadata": {
                "camera_fps": 1.0 / 5.0,  # 5 seconds per frame
                "speech_sample_rate": 16000,
                "llm_model": "llama2"  # TODO: Get from LLM client
            }
        }
        
        return result
        
    def generate_session_json(self, 
                            person_scores: Dict[str, List[Tuple[float, int, Dict]]], 
                            session_metadata: Dict = None) -> Dict:
        """
        Generate a JSON summary for the entire session.
        
        Args:
            person_scores: Map of person names to their score histories
            session_metadata: Additional session metadata
            
        Returns:
            dict: Session summary JSON
        """
        if session_metadata is None:
            session_metadata = {}
            
        # Calculate session duration
        session_duration = time.time() - self.session_start_time
        
        # Summarize each person's scores
        person_summaries = {}
        
        for person_name, scores in person_scores.items():
            if not scores:
                continue
                
            # Calculate statistics
            score_values = [s[1] for s in scores]
            avg_score = sum(score_values) / len(score_values) if score_values else 0
            max_score = max(score_values) if score_values else 0
            min_score = min(score_values) if score_values else 0
            
            # Get most recent score
            latest_timestamp, latest_score, latest_details = scores[-1]
            
            # Create person summary
            person_summaries[person_name] = {
                "average_score": round(avg_score, 1),
                "max_score": max_score,
                "min_score": min_score,
                "data_points": len(scores),
                "latest_score": latest_score,
                "latest_timestamp": datetime.fromtimestamp(latest_timestamp).isoformat() + "Z"
            }
            
        # Construct session JSON
        result = {
            "session_id": self.session_id,
            "start_time": datetime.fromtimestamp(self.session_start_time).isoformat() + "Z",
            "end_time": datetime.fromtimestamp(time.time()).isoformat() + "Z",
            "duration_seconds": round(session_duration, 1),
            "participants": list(person_scores.keys()),
            "participant_summaries": person_summaries,
            "metadata": session_metadata
        }
        
        return result
        
    def save_json_to_file(self, data: Dict, filename: str) -> str:
        """
        Save JSON data to a file.
        
        Args:
            data: JSON data to save
            filename: Output filename
            
        Returns:
            str: Full path to the saved file
        """
        import os
        from config import config
        
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.join(config.DATA_DIR, "json")
            os.makedirs(output_dir, exist_ok=True)
            
            # Full output path
            output_path = os.path.join(output_dir, filename)
            
            # Save JSON
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved JSON to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving JSON to {filename}: {str(e)}")
            return None
            
    def generate_filename(self, person_name: str = None, prefix: str = None) -> str:
        """
        Generate a filename for JSON output.
        
        Args:
            person_name: Optional person name to include
            prefix: Optional prefix for the filename
            
        Returns:
            str: Generated filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        parts = []
        if prefix:
            parts.append(prefix)
            
        if person_name:
            parts.append(person_name)
            
        parts.append(timestamp)
        parts.append(self.session_id[:8])
        
        return "_".join(parts) + ".json"