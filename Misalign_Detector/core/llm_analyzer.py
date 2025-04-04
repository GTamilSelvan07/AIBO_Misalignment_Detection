"""
LLM analyzer module for detecting misalignment in conversations.
"""
import os
import time
import json
import requests
import threading
from queue import Queue
from threading import Thread, Lock

from utils.config import Config
from utils.error_handler import get_logger, LLMError, log_exception

logger = get_logger(__name__)

class LLMAnalyzer:
    """Uses Gemma 3.1b to analyze conversations for misalignment."""
    
    def __init__(self, session_dir):
        """
        Initialize the LLM analyzer.
        
        Args:
            session_dir (str): Directory to save session data
        """
        self.session_dir = session_dir
        self.analysis_dir = os.path.join(session_dir, "analysis")
        
        # Ensure directory exists
        os.makedirs(self.analysis_dir, exist_ok=True)
        
        # Analysis state
        self.is_analyzing = False
        self.analysis_thread = None
        self.analysis_queue = Queue()
        self.analysis_results = []
        self.latest_analysis = None
        
        # Thread safety
        self.lock = Lock()
        
        # Last analyzed transcript to avoid re-analyzing the same content
        self.last_analyzed_transcript = ""
        
        # Temporary scores for when LLM analysis is pending
        self.temp_scores = {}
        
        logger.info("Initialized LLM Analyzer with Gemma 3.1b")
    
    def start(self):
        """Start the analysis thread."""
        if self.is_analyzing:
            logger.warning("LLM Analyzer already running")
            return False
        
        self.is_analyzing = True
        self.analysis_thread = Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()
        
        logger.info("Started LLM Analyzer")
        return True
    
    def stop(self):
        """Stop the analysis thread."""
        self.is_analyzing = False
        
        if self.analysis_thread:
            # Add None to queue to signal analysis thread to stop
            self.analysis_queue.put(None)
            self.analysis_thread.join(timeout=5.0)
        
        logger.info("Stopped LLM Analyzer")
    
    def _analysis_loop(self):
        """Main loop for analyzing transcripts."""
        while self.is_analyzing or not self.analysis_queue.empty():
            try:
                # Get transcript from queue
                item = self.analysis_queue.get()
                
                # None signals to end the thread
                if item is None:
                    break
                
                transcript, features = item
                
                # Skip if transcript is too short or unchanged
                if len(transcript.split()) < 10 or transcript == self.last_analyzed_transcript:
                    self.analysis_queue.task_done()
                    continue
                
                # Update last analyzed transcript
                self.last_analyzed_transcript = transcript
                
                # Analyze transcript
                analysis = self._analyze_with_gemma(transcript, features)
                
                if analysis:
                    # Save analysis
                    self._save_analysis(analysis)
                    
                    # Update latest analysis
                    with self.lock:
                        self.latest_analysis = analysis
                
                self.analysis_queue.task_done()
            
            except Exception as e:
                log_exception(logger, e, "Error in LLM analysis loop")
                time.sleep(1.0)
    
    def _analyze_with_gemma(self, transcript, features):
        """
        Analyze transcript using Gemma 3.1b.
        
        Args:
            transcript (str): Conversation transcript
            features (dict): Dict of participant facial features
            
        Returns:
            dict: Analysis results
        """
        try:
            # Build prompt for Gemma
            prompt = self._build_prompt(transcript, features)
            
            # Call Gemma via OLLAMA API
            response = requests.post(
                Config.OLLAMA_URL,
                json={
                    "model": Config.GEMMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.95,
                        "top_k": 40,
                        "num_predict": 512
                    }
                },
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"Gemma API error: {response.text}")
                return None
            
            result = response.json()
            raw_response = result.get("response", "")
            
            # Parse the structured response
            analysis = self._parse_gemma_response(raw_response, transcript)
            
            return analysis
        
        except Exception as e:
            log_exception(logger, e, "Error analyzing with Gemma")
            return None
    
    def _build_prompt(self, transcript, features):
        """
        Build prompt for Gemma.
        
        Args:
            transcript (str): Conversation transcript
            features (dict): Dict of participant facial features
            
        Returns:
            str: Formatted prompt
        """
        prompt = f"""
You are a misalignment detector analyzing conversations for misunderstandings, confusion, or communication breakdowns.

Below is a transcript of a conversation:

CONVERSATION:
{transcript}

PARTICIPANTS FACIAL FEATURES:
"""
        
        # Add facial features/emotions
        for participant_id, participant_features in features.items():
            confusion = participant_features.get('confusion', 0)
            interest = participant_features.get('interest', 0)
            frustration = participant_features.get('frustration', 0)
            understanding = participant_features.get('understanding', 0)
            
            prompt += f"""
{participant_id}:
- Confusion: {confusion:.2f}
- Interest: {interest:.2f}
- Frustration: {frustration:.2f}
- Understanding: {understanding:.2f}
"""
        
        prompt += """
Your task is to:
1. Identify if there are any misunderstandings, confusion, or misalignments in the conversation.
2. Rate the misalignment severity on a scale from 0 to 1 for each participant.
3. Provide a brief explanation of the likely cause of any misalignment.

Return your response in the following JSON format:
{
  "misalignment_detected": true/false,
  "misalignment_scores": {
    "participant1": 0.7,
    "participant2": 0.3
  },
  "cause": "Brief explanation of what caused the misalignment",
  "recommendation": "Suggestion to resolve the misalignment"
}
"""
        
        return prompt
    
    def _parse_gemma_response(self, response_text, transcript):
        """
        Parse Gemma's response into structured analysis.
        
        Args:
            response_text (str): Raw response from Gemma
            transcript (str): Original transcript
            
        Returns:
            dict: Structured analysis
        """
        try:
            # Extract JSON from response
            json_str = ""
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            
            if start_idx >= 0 and end_idx >= 0:
                json_str = response_text[start_idx:end_idx+1]
            
            # If no JSON found, try to parse the whole response
            if not json_str:
                json_str = response_text
            
            try:
                analysis = json.loads(json_str)
                # Make sure required fields are present
                if "misalignment_detected" not in analysis:
                    analysis["misalignment_detected"] = False
                if "misalignment_scores" not in analysis:
                    analysis["misalignment_scores"] = {}
                if "cause" not in analysis:
                    analysis["cause"] = ""
                if "recommendation" not in analysis:
                    analysis["recommendation"] = ""
            except:
                # Use a simple response format if JSON parsing fails
                analysis = {
                    "misalignment_detected": "misalignment" in response_text.lower() or "confusion" in response_text.lower(),
                    "misalignment_scores": {},
                    "cause": response_text[:200] if len(response_text) > 200 else response_text,
                    "recommendation": "",
                    "parsing_error": True
                }
            
            # Add timestamp and transcript
            analysis["timestamp"] = time.time()
            analysis["transcript"] = transcript
            
            return analysis
        
        except Exception as e:
            log_exception(logger, e, "Error parsing Gemma response")
            return {
                "misalignment_detected": False,
                "misalignment_scores": {},
                "cause": "Error parsing Gemma response",
                "recommendation": "",
                "error": str(e),
                "timestamp": time.time(),
                "transcript": transcript
            }
    
    def _save_analysis(self, analysis):
        """
        Save analysis to file.
        
        Args:
            analysis (dict): Analysis results
        """
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(analysis["timestamp"]))
            file_path = os.path.join(self.analysis_dir, f"analysis_{timestamp}.json")
            
            with open(file_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            # Add to analysis results
            self.analysis_results.append(analysis)
            
            # Keep only the most recent 20 analyses
            if len(self.analysis_results) > 20:
                self.analysis_results = self.analysis_results[-20:]
        
        except Exception as e:
            log_exception(logger, e, "Error saving analysis")
    
    def analyze_transcript(self, transcript, features):
        """
        Queue transcript for analysis.
        
        Args:
            transcript (str): Conversation transcript
            features (dict): Dict of participant facial features
        """
        # Queue for analysis
        self.analysis_queue.put((transcript, features))
    
    def get_latest_analysis(self):
        """
        Get the most recent analysis.
        
        Returns:
            dict: Latest analysis results
        """
        with self.lock:
            return self.latest_analysis
    
    def get_misalignment_scores(self):
        """
        Get the latest misalignment scores for all participants.
        
        Returns:
            dict: Participant misalignment scores
        """
        with self.lock:
            if self.latest_analysis and "misalignment_scores" in self.latest_analysis:
                return self.latest_analysis["misalignment_scores"]
            else:
                return self.temp_scores
    
    def set_temp_scores(self, scores):
        """
        Set temporary scores for when LLM analysis is pending.
        
        Args:
            scores (dict): Temporary scores for each participant
        """
        with self.lock:
            self.temp_scores = scores
    
    def manual_analysis(self, context):
        """
        Manually trigger an analysis with specific context.
        
        Args:
            context (str): Context for analysis
            
        Returns:
            dict: Analysis results
        """
        try:
            prompt = f"""
You are a misalignment detector analyzing conversations for misunderstandings, confusion, or communication breakdowns.

CONTEXT:
{context}

Your task is to:
1. Identify if there are any misunderstandings, confusion, or misalignments described in the context.
2. Rate the misalignment severity on a scale from 0 to 1.
3. Provide a brief explanation of the likely cause of any misalignment.

Return your response in the following JSON format:
{{
  "misalignment_detected": true/false,
  "misalignment_score": 0.7,
  "cause": "Brief explanation of what caused the misalignment",
  "recommendation": "Suggestion to resolve the misalignment"
}}
"""
            
            # Call Gemma via OLLAMA API
            response = requests.post(
                Config.OLLAMA_URL,
                json={
                    "model": Config.GEMMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.95,
                        "top_k": 40,
                        "num_predict": 512
                    }
                },
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"Gemma API error: {response.text}")
                return None
            
            result = response.json()
            raw_response = result.get("response", "")
            
            # Extract JSON from response
            json_str = ""
            start_idx = raw_response.find('{')
            end_idx = raw_response.rfind('}')
            
            if start_idx >= 0 and end_idx >= 0:
                json_str = raw_response[start_idx:end_idx+1]
            
            # If no JSON found, try to parse the whole response
            if not json_str:
                json_str = raw_response
            
            try:
                analysis = json.loads(json_str)
            except:
                analysis = {
                    "misalignment_detected": "misalignment" in raw_response.lower() or "confusion" in raw_response.lower(),
                    "misalignment_score": 0.5,
                    "cause": raw_response[:200] if len(raw_response) > 200 else raw_response,
                    "recommendation": "",
                    "parsing_error": True
                }
            
            # Add timestamp and context
            analysis["timestamp"] = time.time()
            analysis["context"] = context
            
            # Save manual analysis
            self._save_analysis(analysis)
            
            return analysis
        
        except Exception as e:
            error_msg = log_exception(logger, e, "Error in manual analysis")
            raise LLMError(error_msg)