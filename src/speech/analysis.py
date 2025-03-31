"""
Speech analysis module for detecting misalignment in transcribed speech.
Analyzes transcribed text using LLM to identify signs of confusion and misunderstanding.
"""
import time
import json
from collections import deque
import threading
import queue
import re
from typing import Dict, List, Optional, Tuple, Union, Any
from loguru import logger

from config import config


class SpeechAnalyzer:
    """
    Analyzes transcribed speech for signs of misalignment and confusion.
    Uses an LLM to detect patterns of misunderstanding in conversation.
    """
    def __init__(self, llm_client=None):
        """
        Initialize the speech analyzer.
        
        Args:
            llm_client: LLM client for text analysis (if None, one will be created)
        """
        self.llm_client = llm_client
        
        # If no LLM client provided, import and create one
        if self.llm_client is None:
            # Import here to avoid circular imports
            from src.llm.ollama_client import OllamaClient
            self.llm_client = OllamaClient()
            
        # Conversation history for each person
        self.conversations = {}  # {person_name: list of utterances}
        
        # Recent scores for each person
        self.score_history = {}  # {person_name: deque of scores}
        self.history_window = config.scoring.smoothing_window
        
        # Misalignment indicators and patterns
        self.misalignment_indicators = [
            r"(?i)I don'?t understand",
            r"(?i)What do you mean",
            r"(?i)I'?m confused",
            r"(?i)That doesn'?t make sense",
            r"(?i)Could you explain",
            r"(?i)I'?m not sure I follow",
            r"(?i)Can you clarify",
            r"(?i)Wait, what\?",
            r"(?i)I'?m lost",
            r"(?i)You lost me",
            r"(?i)Hmm+\?",
            r"(?i)I'?m not getting it",
            r"(?i)That'?s unclear",
            r"(?i)I'?m not following",
            r"(?i)Sorry, I missed that",
            r"(?i)Umm+, ok\?",
            r"(?i)So you'?re saying",
            r"(?i)Let me make sure I understand",
            r"(?i)Just to be clear"
        ]
        
        # Speech pattern indicators of confusion
        self.confusion_speech_patterns = [
            r"(?i)Um+\s",
            r"(?i)Uh+\s",
            r"(?i)Hmm+\s",
            r"(?i)\.\.\.",
            r"(?i)Well, I mean",
            r"(?i)I mean",
            r"(?i)Like, you know",
            r"(?i)Sort of",
            r"(?i)Kind of",
        ]
        
        # Compiled regular expressions for faster matching
        self.misalignment_regexes = [re.compile(pattern) for pattern in self.misalignment_indicators]
        self.confusion_regexes = [re.compile(pattern) for pattern in self.confusion_speech_patterns]
        
        # Async processing
        self.analysis_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.processing_thread = None
        self.is_processing = False
        
    def add_conversation_utterance(self, person_name: str, utterance: str, 
                                  timestamp: float, confidence: float) -> str:
        """
        Add an utterance to a person's conversation history.
        
        Args:
            person_name: Identifier for the person
            utterance: Transcribed speech
            timestamp: Time of the utterance
            confidence: Transcription confidence (0-1)
            
        Returns:
            str: Utterance ID
        """
        if person_name not in self.conversations:
            self.conversations[person_name] = []
            
        # Create utterance record
        utterance_id = f"utterance_{timestamp}_{hash(utterance)}"
        utterance_record = {
            "id": utterance_id,
            "text": utterance,
            "timestamp": timestamp,
            "confidence": confidence,
            "analyzed": False,
            "score": None,
            "details": None
        }
        
        # Add to conversation history
        self.conversations[person_name].append(utterance_record)
        
        # Keep only the last 20 utterances to limit memory usage
        if len(self.conversations[person_name]) > 20:
            self.conversations[person_name] = self.conversations[person_name][-20:]
            
        return utterance_id
        
    def analyze_utterance_sync(self, person_name: str, utterance: str, 
                              context: Optional[List[Dict]] = None) -> Tuple[int, Dict]:
        """
        Analyze an utterance for signs of misalignment (synchronous version).
        
        Args:
            person_name: Identifier for the person
            utterance: Transcribed speech
            context: Optional context utterances
            
        Returns:
            tuple: (misalignment_score, details)
        """
        if not utterance or not utterance.strip():
            return 0, {"score": 0, "explanation": "Empty utterance"}
            
        # First, check for direct indicators of confusion using regex
        misalignment_matches = []
        for i, regex in enumerate(self.misalignment_regexes):
            matches = regex.findall(utterance)
            if matches:
                misalignment_matches.append({
                    "pattern": self.misalignment_indicators[i],
                    "matches": matches
                })
                
        confusion_matches = []
        for i, regex in enumerate(self.confusion_regexes):
            matches = regex.findall(utterance)
            if matches:
                confusion_matches.append({
                    "pattern": self.confusion_speech_patterns[i],
                    "matches": matches
                })
                
        # Calculate a basic score based on regex matches
        regex_score = 0
        
        # Misalignment indicators are stronger signals
        if misalignment_matches:
            total_matches = sum(len(m["matches"]) for m in misalignment_matches)
            regex_score += min(75, total_matches * 25)  # Up to 75 points for direct indicators
            
        # Confusion speech patterns are weaker signals
        if confusion_matches:
            total_matches = sum(len(m["matches"]) for m in confusion_matches)
            regex_score += min(25, total_matches * 5)  # Up to 25 points for speech patterns
            
        # If we have a high regex score, we can return early without using the LLM
        if regex_score >= 75:
            result = {
                "score": regex_score,
                "explanation": "Strong indicators of misalignment detected",
                "misalignment_indicators": misalignment_matches,
                "confusion_patterns": confusion_matches,
                "llm_analysis": None  # No LLM analysis needed
            }
            
            # Update score history
            self._update_score_history(person_name, regex_score)
            
            return regex_score, result
            
        # If we have some regex matches or the utterance is long enough, use LLM
        if regex_score > 0 or len(utterance.split()) > 5:
            # Prepare context if provided
            context_text = ""
            if context and len(context) > 0:
                context_text = "Previous utterances:\n"
                for i, ctx in enumerate(context[-3:]):  # Use up to last 3 utterances
                    speaker = "Other" if i % 2 == 0 else person_name
                    context_text += f"{speaker}: {ctx['text']}\n"
                    
            # Create prompt for LLM
            prompt = f"""You are analyzing a conversation to detect misalignment, confusion, or misunderstanding.

{context_text}

Current utterance from {person_name}: "{utterance}"

Task: Analyze this utterance for signs of confusion, misalignment, or misunderstanding.
1. Look for explicit statements of confusion ("I don't understand", "That's unclear", etc.)
2. Look for questioning phrases that suggest misalignment ("What do you mean?", "Could you explain?")
3. Look for speech patterns that indicate uncertainty (hesitations, filler words, etc.)
4. Consider the context of the conversation

Respond with a JSON object that includes:
1. A misalignment score from 0-100 (0=completely aligned, 100=completely confused)
2. A brief explanation of your reasoning
3. Specific phrases or words that indicate misalignment

JSON format:
```json
{
  "misalignment_score": <0-100>,
  "explanation": "<brief explanation>",
  "indicators": ["<phrase1>", "<phrase2>", ...],
  "confidence": <0-1>
}
```
"""
            
            try:
                # Call LLM for analysis
                response = self.llm_client.complete(prompt)
                
                # Parse JSON from response
                # Find JSON block in response (between ```json and ```)
                json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Try to find JSON object without code block markers
                    json_match = re.search(r'(\{\s*".*?"\s*:.*\})', response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        # Fallback: use the entire response
                        json_str = response
                        
                try:
                    llm_result = json.loads(json_str)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse LLM response as JSON: {response}")
                    llm_result = {
                        "misalignment_score": regex_score,
                        "explanation": "Failed to parse LLM response",
                        "indicators": [],
                        "confidence": 0.3
                    }
                    
                # Extract score
                llm_score = llm_result.get("misalignment_score", 0)
                
                # Combine regex and LLM scores (weighted combination)
                # If LLM confidence is high, trust it more
                llm_confidence = llm_result.get("confidence", 0.5)
                combined_score = int((0.4 * regex_score) + (0.6 * llm_score * llm_confidence))
                
                # Cap at 100
                combined_score = min(100, combined_score)
                
                # Create result
                result = {
                    "score": combined_score,
                    "explanation": llm_result.get("explanation", "No explanation provided"),
                    "misalignment_indicators": misalignment_matches,
                    "confusion_patterns": confusion_matches,
                    "llm_analysis": {
                        "score": llm_score,
                        "indicators": llm_result.get("indicators", []),
                        "confidence": llm_confidence
                    }
                }
                
                # Update score history
                self._update_score_history(person_name, combined_score)
                
                return combined_score, result
                
            except Exception as e:
                logger.error(f"Error in LLM analysis: {str(e)}")
                
                # Fallback to regex score with penalty for failure
                fallback_score = max(0, regex_score - 10)
                
                result = {
                    "score": fallback_score,
                    "explanation": f"LLM analysis failed: {str(e)}",
                    "misalignment_indicators": misalignment_matches,
                    "confusion_patterns": confusion_matches,
                    "llm_analysis": None
                }
                
                # Update score history
                self._update_score_history(person_name, fallback_score)
                
                return fallback_score, result
                
        else:
            # Short utterance with no regex matches
            result = {
                "score": 0,
                "explanation": "No indicators of misalignment detected",
                "misalignment_indicators": [],
                "confusion_patterns": [],
                "llm_analysis": None
            }
            
            # Update score history
            self._update_score_history(person_name, 0)
            
            return 0, result
            
    def _update_score_history(self, person_name: str, score: int):
        """
        Update the score history for a person.
        
        Args:
            person_name: Identifier for the person
            score: Misalignment score
        """
        if person_name not in self.score_history:
            self.score_history[person_name] = deque(maxlen=self.history_window)
            
        self.score_history[person_name].append(score)
        
    def get_smoothed_score(self, person_name: str) -> int:
        """
        Get the smoothed misalignment score for a person.
        
        Args:
            person_name: Identifier for the person
            
        Returns:
            int: Smoothed misalignment score (0-100)
        """
        if person_name not in self.score_history or len(self.score_history[person_name]) == 0:
            return 0
            
        # Calculate weighted average with more recent scores having higher weight
        scores = list(self.score_history[person_name])
        weights = [i+1 for i in range(len(scores))]  # Increasing weights for more recent scores
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        total_weight = sum(weights)
        
        return int(weighted_sum / total_weight)
        
    def start_processing(self):
        """
        Start the async processing thread.
        """
        if self.is_processing:
            logger.warning("Speech analysis processing already running")
            return
            
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        logger.info("Started speech analysis processing")
        
    def _processing_loop(self):
        """
        Loop for processing analysis requests asynchronously.
        """
        while self.is_processing:
            try:
                # Get the next item to process
                item_id, person_name, utterance, context = self.analysis_queue.get(timeout=1.0)
                
                # Analyze the utterance
                score, details = self.analyze_utterance_sync(person_name, utterance, context)
                
                # Add result to result queue
                self.result_queue.put((item_id, score, details))
                
                # Mark task as done
                self.analysis_queue.task_done()
                
            except queue.Empty:
                # No items to process, just continue
                pass
                
            except Exception as e:
                logger.error(f"Error in speech analysis processing loop: {str(e)}")
                
    def stop_processing(self):
        """
        Stop the async processing thread.
        """
        self.is_processing = False
        
        # Wait for processing thread to end
        if self.processing_thread is not None and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
            
        logger.info("Stopped speech analysis processing")
        
    def analyze_utterance_async(self, person_name: str, utterance: str, 
                              context: Optional[List[Dict]] = None) -> str:
        """
        Queue an utterance for asynchronous analysis.
        
        Args:
            person_name: Identifier for the person
            utterance: Transcribed speech
            context: Optional context utterances
            
        Returns:
            str: Item ID for retrieving the result
        """
        if not self.is_processing:
            self.start_processing()
            
        # Generate a unique ID for this request
        item_id = f"analysis_{time.time()}_{hash(utterance)}"
        
        # Add to queue
        self.analysis_queue.put((item_id, person_name, utterance, context))
        
        return item_id
        
    def get_analysis_result(self, item_id: str = None, timeout: float = 0.0) -> Tuple[Optional[int], Optional[Dict]]:
        """
        Get an analysis result.
        
        Args:
            item_id: Item ID to wait for (None to get any result)
            timeout: Maximum time to wait in seconds (0 for non-blocking)
            
        Returns:
            tuple: (score, details) or (None, None) if no result is available
        """
        try:
            # If item_id is None, get any result
            if item_id is None:
                result_id, score, details = self.result_queue.get(timeout=timeout)
                return score, details
                
            # Otherwise, look for a specific item
            if timeout <= 0:
                # Non-blocking check
                items = []
                while not self.result_queue.empty():
                    try:
                        items.append(self.result_queue.get_nowait())
                        self.result_queue.task_done()
                    except queue.Empty:
                        break
                        
                # Look for our item
                for result_id, score, details in items:
                    if result_id == item_id:
                        # Put other items back
                        for item in items:
                            if item[0] != item_id:
                                self.result_queue.put(item)
                                
                        return score, details
                        
                # Put all items back
                for item in items:
                    self.result_queue.put(item)
                    
                return None, None
                
            else:
                # Blocking check with timeout
                end_time = time.time() + timeout
                
                while time.time() < end_time:
                    # Try to get all items
                    items = []
                    remaining_timeout = end_time - time.time()
                    
                    try:
                        items.append(self.result_queue.get(timeout=remaining_timeout))
                        self.result_queue.task_done()
                    except queue.Empty:
                        break
                        
                    # Get any additional items that are ready
                    while not self.result_queue.empty():
                        try:
                            items.append(self.result_queue.get_nowait())
                            self.result_queue.task_done()
                        except queue.Empty:
                            break
                            
                    # Look for our item
                    for result_id, score, details in items:
                        if result_id == item_id:
                            # Put other items back
                            for item in items:
                                if item[0] != item_id:
                                    self.result_queue.put(item)
                                    
                            return score, details
                            
                    # Put all items back
                    for item in items:
                        self.result_queue.put(item)
                        
                    # Short sleep to prevent tight loop
                    time.sleep(0.1)
                    
                # Timeout occurred
                return None, None
                
        except queue.Empty:
            return None, None
            
    def process_new_utterance(self, person_name: str, utterance: str, 
                            timestamp: float, confidence: float) -> Tuple[str, str]:
        """
        Process a new utterance for a person.
        This method both adds to the conversation history and queues for analysis.
        
        Args:
            person_name: Identifier for the person
            utterance: Transcribed speech
            timestamp: Time of the utterance
            confidence: Transcription confidence (0-1)
            
        Returns:
            tuple: (utterance_id, analysis_id)
        """
        # Add to conversation history
        utterance_id = self.add_conversation_utterance(
            person_name, utterance, timestamp, confidence
        )
        
        # Get last few utterances for context
        context = None
        if person_name in self.conversations and len(self.conversations[person_name]) > 1:
            context = self.conversations[person_name][:-1]  # All except current
            if len(context) > 5:  # Limit context to last 5 utterances
                context = context[-5:]
                
        # Queue for analysis
        analysis_id = self.analyze_utterance_async(person_name, utterance, context)
        
        # Update utterance record with analysis ID
        if person_name in self.conversations:
            for i, utt in enumerate(self.conversations[person_name]):
                if utt["id"] == utterance_id:
                    self.conversations[person_name][i]["analysis_id"] = analysis_id
                    break
                    
        return utterance_id, analysis_id
        
    def get_conversation_summary(self, person_name: str) -> Dict:
        """
        Get a summary of the conversation with a person.
        
        Args:
            person_name: Identifier for the person
            
        Returns:
            dict: Conversation summary
        """
        if person_name not in self.conversations:
            return {
                "person_name": person_name,
                "utterance_count": 0,
                "analyzed_count": 0,
                "current_score": 0,
                "utterances": []
            }
            
        utterances = self.conversations[person_name]
        analyzed_count = sum(1 for u in utterances if u.get("analyzed", False))
        
        return {
            "person_name": person_name,
            "utterance_count": len(utterances),
            "analyzed_count": analyzed_count,
            "current_score": self.get_smoothed_score(person_name),
            "utterances": utterances
        }
        
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
        
    def clear_conversation(self, person_name: str = None):
        """
        Clear conversation history for one or all people.
        
        Args:
            person_name: Specific person to clear, or None for all
        """
        if person_name is None:
            # Clear all conversations
            self.conversations = {}
            logger.info("Cleared all conversation histories")
        elif person_name in self.conversations:
            # Clear specific person
            self.conversations[person_name] = []
            logger.info(f"Cleared conversation history for {person_name}")
            
    def clear_history(self, person_name: str = None):
        """
        Clear score history for one or all people.
        
        Args:
            person_name: Specific person to clear, or None for all
        """
        if person_name is None:
            # Clear all histories
            self.score_history = {}
            logger.info("Cleared all score histories")
        elif person_name in self.score_history:
            # Clear specific person
            self.score_history[person_name] = deque(maxlen=self.history_window)
            logger.info(f"Cleared score history for {person_name}")