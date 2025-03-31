"""
Audio recording module for the misalignment detection system.
Handles microphone input and continuous audio recording.
"""
import os
import time
import queue
import threading
import numpy as np
import wave
import sounddevice as sd
from datetime import datetime
from loguru import logger
from typing import Dict, List, Optional, Tuple, Union, BinaryIO

from config import config


class AudioRecorder:
    """
    Records audio from microphones in timed chunks.
    """
    def __init__(self, device_indexes: List[int] = None, device_names: List[str] = None):
        """
        Initialize the audio recorder.
        
        Args:
            device_indexes: List of audio device indexes to record from
            device_names: Optional list of names for the devices (e.g., "Person A", "Person B")
        """
        self.sample_rate = config.speech.sample_rate
        self.channels = config.speech.channels
        self.recording_interval = config.speech.recording_interval
        
        # Set up device information
        if device_indexes is None:
            # Get default device
            if config.speech.audio_device_index is not None:
                device_indexes = [config.speech.audio_device_index]
            else:
                device_indexes = [None]  # Use system default
                
        if device_names is None:
            device_names = [f"Person_{chr(65+i)}" for i in range(len(device_indexes))]
            
        # Map devices to names
        self.devices = {}
        for i, device_index in enumerate(device_indexes):
            name = device_names[i] if i < len(device_names) else f"Device_{i}"
            self.devices[name] = {
                "index": device_index,
                "stream": None,
                "is_recording": False,
                "audio_queue": queue.Queue(maxsize=100),
                "error_count": 0,
                "last_activity": 0,
                "energy_threshold": config.speech.energy_threshold
            }
            
        # Continuous recording
        self.is_continuous_recording = False
        self.continuous_recording_thread = None
        
        # Thread for saving recordings
        self.saving_thread = None
        self.save_queue = queue.Queue()
        self.is_saving = False
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
    def list_audio_devices(self) -> List[Dict]:
        """
        List available audio devices.
        
        Returns:
            list: List of dictionaries with device information
        """
        try:
            devices = sd.query_devices()
            return devices
        except Exception as e:
            logger.error(f"Error listing audio devices: {str(e)}")
            return []
            
    def start_device(self, device_name: str) -> bool:
        """
        Start recording from a specific device.
        
        Args:
            device_name: Name of the device to start
            
        Returns:
            bool: True if the device was successfully started, False otherwise
        """
        if device_name not in self.devices:
            logger.error(f"Device {device_name} not found")
            return False
            
        device = self.devices[device_name]
        
        if device["is_recording"]:
            logger.warning(f"{device_name}: Already recording")
            return True
            
        try:
            # Define callback function for this device
            def audio_callback(indata, frames, time_info, status):
                """Callback function for the audio stream."""
                if status:
                    logger.warning(f"{device_name} audio status: {status}")
                    
                # Put audio data in the queue
                try:
                    # Calculate audio energy
                    energy = np.mean(np.abs(indata)) * 1000
                    
                    # Update last activity if above threshold
                    if energy > device["energy_threshold"]:
                        device["last_activity"] = time.time()
                        
                    # Add to queue
                    device["audio_queue"].put((indata.copy(), energy, time.time()))
                except queue.Full:
                    # If queue is full, remove oldest item
                    try:
                        device["audio_queue"].get_nowait()
                        device["audio_queue"].put((indata.copy(), energy, time.time()))
                    except (queue.Empty, queue.Full):
                        pass  # Rare race condition, just continue
                        
            # Start the stream
            device["stream"] = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                device=device["index"],
                callback=audio_callback,
                blocksize=int(self.sample_rate * 0.1)  # 100ms blocks for low latency
            )
            
            device["stream"].start()
            device["is_recording"] = True
            device["error_count"] = 0
            logger.info(f"Started recording from {device_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting device {device_name}: {str(e)}")
            device["error_count"] += 1
            return False
            
    def stop_device(self, device_name: str):
        """
        Stop recording from a specific device.
        
        Args:
            device_name: Name of the device to stop
        """
        if device_name not in self.devices:
            logger.error(f"Device {device_name} not found")
            return
            
        device = self.devices[device_name]
        
        if not device["is_recording"]:
            logger.warning(f"{device_name}: Not recording")
            return
            
        try:
            if device["stream"] is not None:
                device["stream"].stop()
                device["stream"].close()
                device["stream"] = None
                
            device["is_recording"] = False
            
            # Clear the queue
            while not device["audio_queue"].empty():
                try:
                    device["audio_queue"].get_nowait()
                except queue.Empty:
                    break
                    
            logger.info(f"Stopped recording from {device_name}")
            
        except Exception as e:
            logger.error(f"Error stopping device {device_name}: {str(e)}")
            
    def start_all_devices(self) -> Dict[str, bool]:
        """
        Start recording from all devices.
        
        Returns:
            dict: Map of device names to success status
        """
        results = {}
        for device_name in self.devices:
            results[device_name] = self.start_device(device_name)
        return results
        
    def stop_all_devices(self):
        """
        Stop recording from all devices.
        """
        for device_name in self.devices:
            self.stop_device(device_name)
            
    def get_audio_chunk(self, device_name: str, duration: float = None) -> Tuple[Optional[np.ndarray], float, float]:
        """
        Get the most recent audio chunk from a device.
        
        Args:
            device_name: Name of the device
            duration: Duration of audio to collect (seconds)
            
        Returns:
            tuple: (audio_data, energy, timestamp) or (None, 0, 0) if no audio is available
        """
        if device_name not in self.devices:
            logger.error(f"Device {device_name} not found")
            return None, 0, 0
            
        device = self.devices[device_name]
        
        if not device["is_recording"]:
            logger.warning(f"{device_name}: Not recording")
            return None, 0, 0
            
        # If no duration specified, get the latest chunk
        if duration is None:
            try:
                return device["audio_queue"].get(timeout=0.1)
            except queue.Empty:
                return None, 0, 0
                
        # Otherwise, collect chunks for the specified duration
        chunks = []
        energies = []
        timestamps = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                audio, energy, timestamp = device["audio_queue"].get(timeout=0.1)
                chunks.append(audio)
                energies.append(energy)
                timestamps.append(timestamp)
            except queue.Empty:
                if len(chunks) > 0:
                    # If we have some data but queue is empty, that's ok
                    break
                else:
                    # If we got nothing, return nothing
                    return None, 0, 0
                    
        if len(chunks) == 0:
            return None, 0, 0
            
        # Concatenate all chunks
        audio_data = np.vstack(chunks)
        avg_energy = np.mean(energies)
        earliest_timestamp = min(timestamps)
        
        return audio_data, avg_energy, earliest_timestamp
        
    def start_continuous_recording(self, interval: float = None) -> bool:
        """
        Start continuously recording audio chunks from all devices.
        
        Args:
            interval: Recording interval in seconds (overrides config if provided)
            
        Returns:
            bool: True if continuous recording was started, False otherwise
        """
        if self.is_continuous_recording:
            logger.warning("Continuous recording already running")
            return True
            
        # Set the recording interval
        if interval is not None:
            self.recording_interval = interval
        else:
            self.recording_interval = config.speech.recording_interval
            
        # Check if we have working devices
        device_status = {}
        for device_name in self.devices:
            if not self.devices[device_name]["is_recording"]:
                success = self.start_device(device_name)
                device_status[device_name] = success
            else:
                device_status[device_name] = True
                
        if not any(device_status.values()):
            logger.error("No working audio devices available for continuous recording")
            return False
            
        # Start the continuous recording thread
        self.is_continuous_recording = True
        self.continuous_recording_thread = threading.Thread(
            target=self._continuous_recording_loop, 
            daemon=True
        )
        self.continuous_recording_thread.start()
        
        # Start the saving thread if not already running
        if not self.is_saving:
            self.is_saving = True
            self.saving_thread = threading.Thread(target=self._saving_loop, daemon=True)
            self.saving_thread.start()
            
        logger.info(f"Started continuous recording with interval {self.recording_interval} seconds")
        return True
        
    def _continuous_recording_loop(self):
        """
        Loop that continuously records audio chunks from all devices.
        """
        next_recording_time = time.time()
        
        while self.is_continuous_recording:
            current_time = time.time()
            
            # Check if it's time to record
            if current_time >= next_recording_time:
                # Record from all devices
                recordings = {}
                for device_name, device in self.devices.items():
                    if device["is_recording"]:
                        # Get audio for the recording interval
                        audio, energy, timestamp = self.get_audio_chunk(
                            device_name, 
                            duration=self.recording_interval
                        )
                        
                        if audio is not None:
                            recordings[device_name] = (audio, energy, timestamp)
                            
                # Calculate next recording time
                next_recording_time = current_time + self.recording_interval
                
                # Notify of recordings
                if recordings:
                    logger.debug(f"Recorded audio from {len(recordings)} devices")
                    
                    # Process recordings (e.g., for transcription)
                    self._process_recordings(recordings)
                    
            # Short sleep to prevent tight loop
            time.sleep(min(0.1, max(0.001, next_recording_time - time.time())))
            
    def _process_recordings(self, recordings: Dict[str, Tuple[np.ndarray, float, float]]):
        """
        Process the recorded audio chunks.
        This method can be overridden in a subclass to implement custom processing.
        
        Args:
            recordings: Dictionary of {device_name: (audio_data, energy, timestamp)}
        """
        # Default implementation: Save recordings with sufficient energy
        for device_name, (audio, energy, timestamp) in recordings.items():
            if energy > self.devices[device_name]["energy_threshold"]:
                # Save audio with timestamp
                timestamp_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
                filename = f"{device_name}_{timestamp_str}.wav"
                
                # Queue for saving
                self.save_queue.put((filename, audio, self.sample_rate, self.channels))
                
    def _saving_loop(self):
        """
        Loop for saving audio recordings in a separate thread.
        """
        while self.is_saving:
            try:
                # Get the next item to save
                filename, audio, sample_rate, channels = self.save_queue.get(timeout=1.0)
                
                # Save the audio
                self.save_audio(filename, audio, sample_rate, channels)
                
                # Mark task as done
                self.save_queue.task_done()
                
            except queue.Empty:
                # No items to save, just continue
                pass
                
            except Exception as e:
                logger.error(f"Error in saving loop: {str(e)}")
                
    def stop_continuous_recording(self):
        """
        Stop continuous recording.
        """
        self.is_continuous_recording = False
        
        # Wait for continuous recording thread to end
        if self.continuous_recording_thread is not None and self.continuous_recording_thread.is_alive():
            self.continuous_recording_thread.join(timeout=1.0)
            
        logger.info("Stopped continuous recording")
        
    def save_audio(self, filename: str, audio: np.ndarray, sample_rate: int, channels: int):
        """
        Save audio data to a WAV file.
        
        Args:
            filename: Output filename
            audio: Audio data as numpy array
            sample_rate: Sample rate of the audio
            channels: Number of audio channels
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.join(config.DATA_DIR, "audio")
            os.makedirs(output_dir, exist_ok=True)
            
            # Full output path
            output_path = os.path.join(output_dir, filename)
            
            # Save as WAV file
            with wave.open(output_path, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                
                # Convert to int16
                audio_int16 = (audio * 32767).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())
                
            logger.debug(f"Saved audio to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving audio {filename}: {str(e)}")
            return None
            
    def set_recording_interval(self, interval: float):
        """
        Set the recording interval for continuous recording.
        
        Args:
            interval: Recording interval in seconds
        """
        if interval < 0.1:
            logger.warning(f"Recording interval {interval} too small, setting to 0.1 seconds")
            interval = 0.1
            
        self.recording_interval = interval
        logger.info(f"Set recording interval to {interval} seconds")
        
    def is_device_active(self, device_name: str, timeout: float = 5.0) -> bool:
        """
        Check if a device has recently detected speech.
        
        Args:
            device_name: Name of the device
            timeout: Time window to check for activity (seconds)
            
        Returns:
            bool: True if the device has detected speech within the timeout period
        """
        if device_name not in self.devices:
            return False
            
        device = self.devices[device_name]
        
        if not device["is_recording"]:
            return False
            
        # Check if there's been activity within the timeout period
        return time.time() - device["last_activity"] < timeout
        
    def get_device_status(self) -> Dict[str, Dict]:
        """
        Get the status of all devices.
        
        Returns:
            dict: Map of device names to status information
        """
        status = {}
        for name, device in self.devices.items():
            status[name] = {
                "is_recording": device["is_recording"],
                "error_count": device["error_count"],
                "active": self.is_device_active(name),
                "energy_threshold": device["energy_threshold"]
            }
        return status
        
    def adjust_energy_threshold(self, device_name: str, threshold: int):
        """
        Adjust the energy threshold for speech detection.
        
        Args:
            device_name: Name of the device
            threshold: New energy threshold
        """
        if device_name not in self.devices:
            logger.error(f"Device {device_name} not found")
            return
            
        self.devices[device_name]["energy_threshold"] = threshold
        logger.info(f"Adjusted energy threshold for {device_name} to {threshold}")
        
    def get_active_devices(self) -> List[str]:
        """
        Get a list of currently active devices.
        
        Returns:
            list: Names of active devices
        """
        return [name for name, device in self.devices.items() 
                if device["is_recording"] and self.is_device_active(name)]
                
    def calibrate_device(self, device_name: str, duration: float = 5.0) -> bool:
        """
        Calibrate the energy threshold for a device based on ambient noise.
        
        Args:
            device_name: Name of the device
            duration: Duration of calibration in seconds
            
        Returns:
            bool: True if calibration was successful
        """
        if device_name not in self.devices:
            logger.error(f"Device {device_name} not found")
            return False
            
        device = self.devices[device_name]
        
        if not device["is_recording"]:
            if not self.start_device(device_name):
                return False
                
        try:
            logger.info(f"Calibrating {device_name} for {duration} seconds...")
            
            # Collect audio for the specified duration
            audio, energy, _ = self.get_audio_chunk(device_name, duration)
            
            if audio is None:
                logger.error(f"No audio collected during calibration for {device_name}")
                return False
                
            # Calculate new threshold based on noise level
            energy_values = []
            for i in range(0, len(audio), int(self.sample_rate * 0.1)):
                chunk = audio[i:i+int(self.sample_rate * 0.1)]
                if len(chunk) > 0:
                    energy_values.append(np.mean(np.abs(chunk)) * 1000)
                    
            if not energy_values:
                logger.error(f"No energy values calculated during calibration for {device_name}")
                return False
                
            # Set threshold to mean + 2 standard deviations
            mean_energy = np.mean(energy_values)
            std_energy = np.std(energy_values)
            new_threshold = int(mean_energy + 2 * std_energy)
            
            # Update threshold
            self.adjust_energy_threshold(device_name, new_threshold)
            
            logger.info(f"Calibrated {device_name}: mean={mean_energy:.1f}, std={std_energy:.1f}, threshold={new_threshold}")
            return True
            
        except Exception as e:
            logger.error(f"Error calibrating {device_name}: {str(e)}")
            return False