"""
Speech-to-text transcription module for the misalignment detection system.
"""
import os
import time
import threading
import queue
import numpy as np
import whisper
import speech_recognition as sr
from typing import Dict, List, Optional, Tuple, Union
from loguru import logger

from config import config


class SpeechTranscriber:
    """
    Transcribes audio to text using either Whisper or SpeechRecognition.
    """
    def __init__(self, use_whisper: bool = True):
        """
        Initialize the speech transcriber.
        
        Args:
            use_whisper: Whether to use OpenAI's Whisper model (otherwise uses SpeechRecognition)
        """
        self.use_whisper = use_whisper
        self.whisper_model = None
        self.recognizer = sr.Recognizer()
        self.sample_rate = config.speech.sample_rate
        
        # Adjust speech recognition parameters
        self.recognizer.energy_threshold = config.speech.energy_threshold
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = config.speech.phrase_threshold
        
        # Transcription queue and processing thread
        self.transcription_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.processing_thread = None
        self.is_processing = False
        
        # Initialize whisper in a separate thread
        if self.use_whisper:
            self.whisper_initialization_thread = threading.Thread(
                target=self._initialize_whisper, 
                daemon=True
            )
            self.whisper_initialization_thread.start()
        else:
            self.whisper_initialization_thread = None
            
    def _initialize_whisper(self):
        """
        Initialize the Whisper model.
        """
        try:
            logger.info(f"Initializing Whisper model '{config.speech.whisper_model}'...")
            start_time = time.time()
            
            # Load the whisper model
            self.whisper_model = whisper.load_model(config.speech.whisper_model)
            
            logger.info(f"Whisper model initialized in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error initializing Whisper model: {str(e)}")
            self.use_whisper = False
            
    def wait_for_whisper_initialization(self, timeout: float = 30.0) -> bool:
        """
        Wait for the Whisper model to initialize.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        if not self.use_whisper:
            return False
            
        if self.whisper_model is not None:
            return True
            
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.whisper_model is not None:
                return True
            time.sleep(0.1)
            
        # Timeout occurred
        logger.error(f"Whisper model initialization timed out after {timeout} seconds")
        return False
        
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = None) -> Tuple[str, float]:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            tuple: (transcription, confidence)
        """
        if audio_data is None or len(audio_data) == 0:
            return "", 0.0
            
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        # Check if audio has enough energy
        energy = np.mean(np.abs(audio_data)) * 1000
        if energy < config.speech.energy_threshold:
            return "", 0.0
            
        try:
            if self.use_whisper and self.whisper_model is not None:
                # Use Whisper for transcription
                result = self.whisper_model.transcribe(
                    audio_data, 
                    fp16=False,
                    language="en"  # Can be configured for other languages
                )
                
                text = result["text"].strip()
                confidence = result.get("confidence", 0.7)  # Default confidence if not provided
                
                return text, confidence
                
            else:
                # Use SpeechRecognition for transcription
                # Convert numpy array to AudioData
                audio = sr.AudioData(
                    (audio_data * 32767).astype(np.int16).tobytes(),
                    sample_rate,
                    2  # 16-bit
                )
                
                # Recognize speech using Google Speech Recognition
                try:
                    text = self.recognizer.recognize_google(audio, show_all=True)
                    
                    if isinstance(text, dict) and "alternative" in text:
                        best_result = text["alternative"][0]
                        text = best_result["transcript"]
                        confidence = best_result.get("confidence", 0.0)
                        
                        return text, confidence
                    elif isinstance(text, list) and len(text) > 0:
                        return text[0], 0.6  # Default confidence
                    else:
                        return "", 0.0
                        
                except sr.UnknownValueError:
                    return "", 0.0
                except sr.RequestError as e:
                    logger.error(f"Google Speech Recognition service error: {str(e)}")
                    return "", 0.0
                    
        except Exception as e:
            logger.error(f"Error in transcription: {str(e)}")
            return "", 0.0
            
    def start_processing(self):
        """
        Start the transcription processing thread.
        """
        if self.is_processing:
            logger.warning("Transcription processing already running")
            return
            
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        logger.info("Started transcription processing")
        
    def _processing_loop(self):
        """
        Loop for processing transcription requests.
        """
        while self.is_processing:
            try:
                # Get the next item to process
                item_id, audio_data, sample_rate, metadata = self.transcription_queue.get(timeout=1.0)
                
                # Transcribe the audio
                text, confidence = self.transcribe_audio(audio_data, sample_rate)
                
                # Add result to result queue
                self.result_queue.put((item_id, text, confidence, metadata))
                
                # Mark task as done
                self.transcription_queue.task_done()
                
            except queue.Empty:
                # No items to process, just continue
                pass
                
            except Exception as e:
                logger.error(f"Error in transcription processing loop: {str(e)}")
                
    def stop_processing(self):
        """
        Stop the transcription processing thread.
        """
        self.is_processing = False
        
        # Wait for processing thread to end
        if self.processing_thread is not None and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
            
        logger.info("Stopped transcription processing")
        
    def queue_audio_for_transcription(self, audio_data: np.ndarray, 
                                     sample_rate: int = None,
                                     metadata: Dict = None) -> str:
        """
        Queue audio for asynchronous transcription.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            metadata: Additional metadata to include with the result
            
        Returns:
            str: Item ID for retrieving the result
        """
        if not self.is_processing:
            self.start_processing()
            
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        if metadata is None:
            metadata = {}
            
        # Generate a unique ID for this request
        item_id = f"transcription_{time.time()}_{hash(str(audio_data))}"
        
        # Add to queue
        self.transcription_queue.put((item_id, audio_data, sample_rate, metadata))
        
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
            
    def transcribe_file(self, filepath: str) -> Tuple[str, float]:
        """
        Transcribe an audio file.
        
        Args:
            filepath: Path to the audio file
            
        Returns:
            tuple: (transcription, confidence)
        """
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return "", 0.0
            
        try:
            if self.use_whisper and self.whisper_model is not None:
                # Use Whisper for transcription
                result = self.whisper_model.transcribe(filepath)
                
                text = result["text"].strip()
                confidence = result.get("confidence", 0.7)  # Default confidence if not provided
                
                return text, confidence
                
            else:
                # Use SpeechRecognition for transcription
                with sr.AudioFile(filepath) as source:
                    audio = self.recognizer.record(source)
                    
                try:
                    text = self.recognizer.recognize_google(audio, show_all=True)
                    
                    if isinstance(text, dict) and "alternative" in text:
                        best_result = text["alternative"][0]
                        text = best_result["transcript"]
                        confidence = best_result.get("confidence", 0.0)
                        
                        return text, confidence
                    elif isinstance(text, list) and len(text) > 0:
                        return text[0], 0.6  # Default confidence
                    else:
                        return "", 0.0
                        
                except sr.UnknownValueError:
                    return "", 0.0
                except sr.RequestError as e:
                    logger.error(f"Google Speech Recognition service error: {str(e)}")
                    return "", 0.0
                    
        except Exception as e:
            logger.error(f"Error transcribing file {filepath}: {str(e)}")
            return "", 0.0