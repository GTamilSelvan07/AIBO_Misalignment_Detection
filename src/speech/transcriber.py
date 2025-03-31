"""
Speech transcription module for the misalignment detection system.
Converts recorded audio to text using Whisper or other speech recognition.
"""
import time
import threading
import queue
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger

from config import config, MODELS_DIR

# Try to import whisper, fall back to SpeechRecognition if not available
WHISPER_AVAILABLE = False
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    logger.warning("whisper package not found. Falling back to SpeechRecognition.")
    logger.warning("To install whisper, run: pip install openai-whisper")
    try:
        import speech_recognition as sr
    except ImportError:
        logger.error("SpeechRecognition package not found. Speech transcription disabled.")
        logger.error("To install SpeechRecognition, run: pip install SpeechRecognition")


class SpeechTranscriber:
    """
    Transcribes audio to text using Whisper or SpeechRecognition.
    """
    def __init__(self, whisper_model: str = None):
        """
        Initialize the speech transcriber.
        
        Args:
            whisper_model: Whisper model name to use (tiny, base, small, medium, large)
        """
        self.whisper_model_name = whisper_model or config.speech.whisper_model
        self.use_whisper = WHISPER_AVAILABLE
        
        # Transcription components
        self.whisper_model = None
        self.recognizer = None  # For SpeechRecognition fallback
        
        # Processing queue
        self.transcription_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_processing = False
        self.processing_thread = None
        
        # Initialization
        self.is_initialized = False
        self.initialization_thread = threading.Thread(target=self._initialize_transcriber, daemon=True)
        self.initialization_thread.start()
        
    def _initialize_transcriber(self):
        """
        Initialize the transcription system based on available libraries.
        """
        try:
            if self.use_whisper:
                self._initialize_whisper()
            else:
                self._initialize_speech_recognition()
                
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Error initializing speech transcriber: {str(e)}")
            self.is_initialized = False
            
    def _initialize_whisper(self):
        """
        Initialize the Whisper model.
        """
        logger.info(f"Initializing Whisper model '{self.whisper_model_name}'...")
        start_time = time.time()
        
        try:
            # Load the Whisper model
            self.whisper_model = whisper.load_model(self.whisper_model_name)
            logger.info(f"Whisper model initialized in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {str(e)}")
            self.use_whisper = False
            self._initialize_speech_recognition()
            
    def _initialize_speech_recognition(self):
        """
        Initialize SpeechRecognition as a fallback.
        """
        logger.info("Initializing SpeechRecognition...")
        
        try:
            self.recognizer = sr.Recognizer()
            logger.info("SpeechRecognition initialized")
        except Exception as e:
            logger.error(f"Error initializing SpeechRecognition: {str(e)}")
            
    def wait_for_initialization(self, timeout: float = 30.0) -> bool:
        """
        Wait for the transcriber to initialize.
        
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
            time.sleep(0.1)
            
        # Timeout occurred
        logger.error(f"Speech transcriber initialization timed out after {timeout} seconds")
        return False
        
    def start_processing(self):
        """
        Start the processing thread for transcription.
        """
        if self.is_processing:
            logger.warning("Speech transcription already running")
            return
            
        # Wait for initialization
        if not self.wait_for_initialization(timeout=30.0):
            logger.error("Cannot start processing: transcriber not initialized")
            return
            
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        logger.info("Started speech transcription processing")
        
    def _processing_loop(self):
        """
        Main loop for processing audio transcription requests.
        """
        while self.is_processing:
            try:
                # Get the next item to process
                item_id, audio, metadata = self.transcription_queue.get(timeout=1.0)
                
                # Process the audio
                text, confidence = self._transcribe_audio(audio)
                
                # Add the result to the result queue
                self.result_queue.put((item_id, text, confidence, metadata))
                
                # Mark task as done
                self.transcription_queue.task_done()
                
            except queue.Empty:
                # No items to process, just continue
                pass
                
            except Exception as e:
                logger.error(f"Error in speech transcription processing loop: {str(e)}")
                
    def _transcribe_audio(self, audio: np.ndarray) -> Tuple[str, float]:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            tuple: (transcribed_text, confidence)
        """
        if audio is None or len(audio) == 0:
            return "", 0.0
            
        try:
            if self.use_whisper and self.whisper_model is not None:
                return self._transcribe_with_whisper(audio)
            elif self.recognizer is not None:
                return self._transcribe_with_speech_recognition(audio)
            else:
                return "", 0.0
                
        except Exception as e:
            logger.error(f"Error in audio transcription: {str(e)}")
            return "", 0.0
            
    def _transcribe_with_whisper(self, audio: np.ndarray) -> Tuple[str, float]:
        """
        Transcribe audio using Whisper.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            tuple: (transcribed_text, confidence)
        """
        try:
            # Ensure audio is float32 and in the range [-1, 1]
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32) / 32768.0  # Convert from int16
                
            # Run Whisper transcription
            result = self.whisper_model.transcribe(
                audio,
                language="en",  # Specify language for faster processing
                fp16=False      # Use FP16 for faster processing if available
            )
            
            # Extract text and confidence
            text = result["text"].strip()
            confidence = 0.0
            
            # Extract confidence if available
            if "segments" in result and len(result["segments"]) > 0:
                # Average confidence across segments
                segment_confidences = [seg.get("confidence", 0.0) for seg in result["segments"]]
                if segment_confidences:
                    confidence = sum(segment_confidences) / len(segment_confidences)
            
            return text, confidence
            
        except Exception as e:
            logger.error(f"Error in Whisper transcription: {str(e)}")
            return "", 0.0
            
    def _transcribe_with_speech_recognition(self, audio: np.ndarray) -> Tuple[str, float]:
        """
        Transcribe audio using SpeechRecognition.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            tuple: (transcribed_text, confidence)
        """
        try:
            # Convert numpy array to AudioData
            audio_int16 = (audio * 32767).astype(np.int16)
            sample_rate = config.speech.sample_rate
            
            audio_data = sr.AudioData(
                audio_int16.tobytes(),
                sample_rate=sample_rate,
                sample_width=2  # 16-bit
            )
            
            # Use Google's speech recognition (can be changed to other providers)
            text = self.recognizer.recognize_google(audio_data)
            confidence = 0.8  # Hardcoded confidence since Google doesn't provide it
            
            return text, confidence
            
        except sr.UnknownValueError:
            logger.debug("Speech Recognition could not understand audio")
            return "", 0.0
            
        except sr.RequestError as e:
            logger.error(f"Could not request results from Speech Recognition service: {e}")
            return "", 0.0
            
        except Exception as e:
            logger.error(f"Error in SpeechRecognition transcription: {str(e)}")
            return "", 0.0
            
    def queue_audio_for_transcription(self, audio: np.ndarray, metadata: Dict = None) -> str:
        """
        Queue audio for transcription.
        
        Args:
            audio: Audio data as numpy array
            metadata: Optional metadata to associate with the transcription
            
        Returns:
            str: Item ID for retrieving the result
        """
        if not self.is_processing:
            self.start_processing()
            
        # Generate a unique ID for this request
        item_id = f"transcription_{time.time()}_{hash(audio.tobytes() if audio is not None else 'None')}"
        
        # Add to queue
        self.transcription_queue.put((item_id, audio, metadata))
        
        return item_id
        
    def get_transcription_result(self, item_id: str = None, timeout: float = 0.0) -> Tuple[Optional[str], Optional[float], Optional[Dict]]:
        """
        Get a transcription result.
        
        Args:
            item_id: Item ID to wait for (None to get any result)
            timeout: Maximum time to wait in seconds (0 for non-blocking)
            
        Returns:
            tuple: (text, confidence, metadata) or (None, None, None) if no result is available
        """
        try:
            # If item_id is None, get any result
            if item_id is None:
                result_id, text, confidence, metadata = self.result_queue.get(timeout=timeout)
                return text, confidence, metadata
                
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
                for result_id, text, confidence, metadata in items:
                    if result_id == item_id:
                        # Put other items back
                        for item in items:
                            if item[0] != item_id:
                                self.result_queue.put(item)
                                
                        return text, confidence, metadata
                        
                # Put all items back
                for item in items:
                    self.result_queue.put(item)
                    
                return None, None, None
                
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
                    for result_id, text, confidence, metadata in items:
                        if result_id == item_id:
                            # Put other items back
                            for item in items:
                                if item[0] != item_id:
                                    self.result_queue.put(item)
                                    
                            return text, confidence, metadata
                            
                    # Put all items back
                    for item in items:
                        self.result_queue.put(item)
                        
                    # Short sleep to prevent tight loop
                    time.sleep(0.1)
                    
                # Timeout occurred
                return None, None, None
                
        except queue.Empty:
            return None, None, None
            
        except Exception as e:
            logger.error(f"Error getting transcription result: {str(e)}")
            return None, None, None
            
    def stop_processing(self):
        """
        Stop the processing thread.
        """
        self.is_processing = False
        
        # Wait for processing thread to end
        if self.processing_thread is not None and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
            
        logger.info("Stopped speech transcription processing")
        
    def get_status(self) -> Dict:
        """
        Get the status of the transcriber.
        
        Returns:
            dict: Status information
        """
        return {
            "initialized": self.is_initialized,
            "processing": self.is_processing,
            "using_whisper": self.use_whisper,
            "queue_size": self.transcription_queue.qsize(),
            "result_queue_size": self.result_queue.qsize()
        }