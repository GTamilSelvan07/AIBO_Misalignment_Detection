"""
Parser for LLM responses in the misalignment detection system.
"""
import json
import re
from typing import Dict, Optional, Any
from loguru import logger


def parse_llm_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Parse a JSON response from the LLM.
    
    Args:
        response: Raw response from the LLM
        
    Returns:
        dict: Parsed JSON object or None if parsing fails
    """
    if not response:
        return None
        
    try:
        # First, try to find JSON block in response (between ```json and ```)
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
            
        # If that fails, try to find a JSON object without code block markers
        json_match = re.search(r'(\{\s*".*?"\s*:.*\})', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
            
        # If that fails, try to parse the entire response as JSON
        return json.loads(response)
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.debug(f"Problematic response: {response}")
        return None
    except Exception as e:
        logger.error(f"Error parsing LLM response: {str(e)}")
        return None
        

def extract_misalignment_score(parsed_response: Optional[Dict[str, Any]]) -> int:
    """
    Extract the misalignment score from a parsed LLM response.
    
    Args:
        parsed_response: Parsed response from parse_llm_response
        
    Returns:
        int: Misalignment score (0-100) or 0 if not found
    """
    if not parsed_response:
        return 0
        
    # Look for different possible key names
    for key in ['misalignment_score', 'score', 'confused_score', 'confusion_score', 'combined_score']:
        if key in parsed_response:
            try:
                score = int(parsed_response[key])
                return max(0, min(100, score))  # Ensure range 0-100
            except (ValueError, TypeError):
                pass
                
    # If no score found, return 0
    return 0
    

def extract_misalignment_details(parsed_response: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract the details of misalignment from a parsed LLM response.
    
    Args:
        parsed_response: Parsed response from parse_llm_response
        
    Returns:
        dict: Details of misalignment
    """
    if not parsed_response:
        return {
            "explanation": "No explanation available",
            "indicators": [],
            "misalignment_type": "unknown",
            "confidence": 0.0
        }
        
    details = {}
    
    # Extract explanation
    for key in ['explanation', 'reason', 'description', 'analysis']:
        if key in parsed_response and parsed_response[key]:
            details['explanation'] = parsed_response[key]
            break
    if 'explanation' not in details:
        details['explanation'] = "No explanation available"
        
    # Extract indicators
    for key in ['indicators', 'phrases', 'markers', 'cues']:
        if key in parsed_response and parsed_response[key]:
            if isinstance(parsed_response[key], list):
                details['indicators'] = parsed_response[key]
                break
            elif isinstance(parsed_response[key], str):
                details['indicators'] = [parsed_response[key]]
                break
    if 'indicators' not in details:
        details['indicators'] = []
        
    # Extract misalignment type
    for key in ['misalignment_type', 'type', 'confusion_type', 'category']:
        if key in parsed_response and parsed_response[key]:
            details['misalignment_type'] = parsed_response[key]
            break
    if 'misalignment_type' not in details:
        details['misalignment_type'] = "unknown"
        
    # Extract confidence
    for key in ['confidence', 'certainty', 'score_confidence']:
        if key in parsed_response and parsed_response[key] is not None:
            try:
                conf = float(parsed_response[key])
                details['confidence'] = max(0.0, min(1.0, conf))  # Ensure range 0.0-1.0
                break
            except (ValueError, TypeError):
                pass
    if 'confidence' not in details:
        details['confidence'] = 0.5  # Default confidence
        
    return details