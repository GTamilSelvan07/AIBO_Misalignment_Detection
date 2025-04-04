"""
Audio manager module for recording audio and transcribing speech.
"""
import os
import time
import json
import wave
import pyaudio
import subprocess
import threading
import numpy as np
from datetime import datetime
from queue import Queue
from threading import Thread, Event
from collections import deque

from utils.config import Config
from utils.error_handler import get_logger, AudioError, log_exception
from utils.helpers import format_timestamp

logger = get_logger(__name__)

class AudioManager:
    """Manages audio recording and transcription."""
    
    def __init__(self, session_dir):
        """
        Initialize the audio manager.
        
        Args:
            session_dir (str): Directory to save session data
        """
        self.session_dir = session_dir
        self.audio_dir = os.path.join(session_dir, "audio")
        self.transcript_dir = os.path.join(session_dir, "transcripts")
        
        # Ensure directories exist
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.transcript_dir, exist_ok=True)
        
        # Audio recording settings
        self.sample_rate = Config.AUDIO_SAMPLE_RATE
        self.channels = Config.AUDIO_CHANNELS
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        
        # Recording state
        self.is_recording = False
        self.is_transcribing = False
        self.record_thread = None
        self.transcribe_thread = None
        self.audio_stream = None
        self.py_audio = None
        
        # File paths
        self.current_audio_file = None
        self.transcript_file = os.path.join(self.transcript_dir, "transcript.json")
        
        # Store audio chunks for the current segment
        self.audio_chunks = []
        
        # Audio segment management
        self.segment_duration = 10  # seconds
        self.segment_samples = self.sample_rate * self.segment_duration
        self.segment_counter = 0
        
        # Queue for audio segments to transcribe
        self.transcription_queue = Queue()
        
        # Store transcribed segments
        self.transcribed_segments = []
        
        # Event to signal new transcription available
        self.new_transcription_event = Event()
        
        # For live transcription updates
        self.current_transcript = ""
        
        logger.info("Initialized Audio Manager")
    
    def start_recording(self):
        """Start audio recording and transcription."""
        if self.is_recording:
            logger.warning("Audio recording already in progress")
            return False
        
        try:
            # Initialize PyAudio
            self.py_audio = pyaudio.PyAudio()
            
            # Open audio stream
            self.audio_stream = self.py_audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            self.is_recording = True
            self.record_thread = Thread(target=self._record_loop, daemon=True)
            self.record_thread.start()
            
            self.is_transcribing = True
            self.transcribe_thread = Thread(target=self._transcribe_loop, daemon=True)
            self.transcribe_thread.start()
            
            logger.info("Started audio recording and transcription")
            return True
        
        except Exception as e:
            error_msg = log_exception(logger, e, "Failed to start audio recording")
            raise AudioError(error_msg)
    
    def stop_recording(self):
        """Stop audio recording and transcription."""
        if not self.is_recording:
            logger.warning("No audio recording in progress")
            return False
        
        self.is_recording = False
        self.is_transcribing = False
        
        # Wait for threads to finish
        if self.record_thread:
            self.record_thread.join(timeout=2.0)
        
        if self.transcribe_thread:
            # Add None to queue to signal transcribe thread to finish processing
            self.transcription_queue.put(None)
            self.transcribe_thread.join(timeout=5.0)
        
        # Close audio stream
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None
        
        # Close PyAudio
        if self.py_audio:
            self.py_audio.terminate()
            self.py_audio = None
        
        # Save any remaining audio
        self._save_current_segment()
        
        logger.info("Stopped audio recording and transcription")
        return True
    
    def _record_loop(self):
        """Main loop for recording audio."""
        start_time = time.time()
        samples_recorded = 0
        
        while self.is_recording:
            try:
                # Read audio chunk
                audio_chunk = self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
                self.audio_chunks.append(audio_chunk)
                
                # Update samples recorded
                samples_recorded += self.chunk_size
                
                # Check if we've recorded enough for a segment
                if samples_recorded >= self.segment_samples:
                    self._save_current_segment()
                    samples_recorded = 0
                    
                # Slight delay to reduce CPU usage
                time.sleep(0.001)
            
            except Exception as e:
                log_exception(logger, e, "Error in audio recording loop")
                time.sleep(0.1)
    
    def _save_current_segment(self):
        """Save the current audio segment to file and queue for transcription."""
        if not self.audio_chunks:
            return
        
        try:
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            segment_file = os.path.join(self.audio_dir, f"segment_{timestamp}_{self.segment_counter}.wav")
            self.segment_counter += 1
            
            # Save audio to WAV file
            with wave.open(segment_file, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.py_audio.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.audio_chunks))
            
            # Clear audio chunks
            self.audio_chunks = []
            
            # Queue file for transcription
            self.transcription_queue.put(segment_file)
            
            logger.info(f"Saved audio segment: {segment_file}")
            
            # Update current audio file
            self.current_audio_file = segment_file
        
        except Exception as e:
            log_exception(logger, e, "Error saving audio segment")
    
    def _transcribe_loop(self):
        """Main loop for transcribing audio segments."""
        while self.is_transcribing or not self.transcription_queue.empty():
            try:
                # Get audio file from queue
                audio_file = self.transcription_queue.get()
                
                # None signals to end the thread
                if audio_file is None:
                    break
                
                # Transcribe audio file
                transcript = self._transcribe_with_whisper(audio_file)
                
                if transcript:
                    # Add to transcribed segments
                    timestamp = time.time()
                    segment = {
                        "timestamp": format_timestamp(timestamp),
                        "unix_timestamp": timestamp,
                        "audio_file": os.path.basename(audio_file),
                        "text": transcript
                    }
                    
                    self.transcribed_segments.append(segment)
                    
                    # Update current transcript
                    self.current_transcript = transcript
                    
                    # Save transcripts to file
                    self._save_transcripts()
                    
                    # Signal new transcription available
                    self.new_transcription_event.set()
                    self.new_transcription_event.clear()
                
                self.transcription_queue.task_done()
            
            except Exception as e:
                log_exception(logger, e, "Error in transcription loop")
                time.sleep(0.5)
    
    def _transcribe_with_whisper(self, audio_file):
        """
        Transcribe audio file using Whisper.
        
        Args:
            audio_file (str): Path to audio file
            
        Returns:
            str: Transcribed text
        """
        try:
            # Build command for Whisper
            cmd = [
                Config.WHISPER_PATH,
                "--model", "small",  # Adjust model size as needed
                "--language", "en",  # Adjust language as needed
                "--output_format", "json",
                "--output_dir", self.transcript_dir,
                audio_file
            ]
            
            # Run Whisper as subprocess
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Whisper error: {stderr}")
                return ""
            
            # Parse the JSON output
            json_file = os.path.join(
                self.transcript_dir, 
                os.path.basename(audio_file).replace('.wav', '.json')
            )
            
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    result = json.load(f)
                    if 'text' in result:
                        return result['text'].strip()
            
            return ""
        
        except Exception as e:
            log_exception(logger, e, f"Error transcribing audio file: {audio_file}")
            return ""
    
    def _save_transcripts(self):
        """Save all transcribed segments to a JSON file."""
        try:
            with open(self.transcript_file, 'w') as f:
                json.dump({
                    "segments": self.transcribed_segments,
                    "updated_at": format_timestamp()
                }, f, indent=2)
        
        except Exception as e:
            log_exception(logger, e, "Error saving transcripts to file")
    
    def get_transcript(self, max_segments=5):
        """
        Get the most recent transcript.
        
        Args:
            max_segments (int): Maximum number of recent segments to include
            
        Returns:
            str: Combined transcript text
        """
        if not self.transcribed_segments:
            return ""
        
        # Get the most recent segments
        recent_segments = self.transcribed_segments[-max_segments:]
        
        # Combine segment text
        transcript = " ".join(segment["text"] for segment in recent_segments)
        
        return transcript
    
    def get_full_transcript(self):
        """
        Get the full transcript history.
        
        Returns:
            list: List of all transcribed segments
        """
        return self.transcribed_segments
    
    def wait_for_new_transcription(self, timeout=None):
        """
        Wait for a new transcription to be available.
        
        Args:
            timeout (float, optional): Maximum time to wait in seconds
            
        Returns:
            bool: True if new transcription is available, False if timeout
        """
        return self.new_transcription_event.wait(timeout)