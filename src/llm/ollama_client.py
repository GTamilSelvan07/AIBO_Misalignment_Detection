"""
OLLAMA API client for LLM integration in the misalignment detection system.
"""
import os
import time
import json
import threading
import requests
from typing import Dict, List, Optional, Union, Any
from loguru import logger

from config import config


class OllamaClient:
    """
    Client for interacting with OLLAMA API.
    
    OLLAMA is a local LLM server that can run various language models.
    Official API docs: https://github.com/ollama/ollama/blob/main/docs/api.md
    """
    def __init__(self, host: str = None, model_name: str = None):
        """
        Initialize the OLLAMA client.
        
        Args:
            host: OLLAMA API host URL
            model_name: Name of the model to use
        """
        self.host = host or config.llm.ollama_host
        self.model_name = model_name or config.llm.model_name
        self.timeout = config.llm.timeout
        
        # Default generation parameters
        self.default_params = {
            "model": self.model_name,
            "max_tokens": config.llm.max_tokens,
            "temperature": config.llm.temperature,
            "top_p": config.llm.top_p,
            "top_k": config.llm.top_k,
            "stream": False
        }
        
        # Connection status
        self.is_connected = False
        self.connection_error = None
        
        # Connection check thread
        self.connection_check_thread = threading.Thread(
            target=self._check_connection, 
            daemon=True
        )
        self.connection_check_thread.start()
        
    def _check_connection(self):
        """
        Check if OLLAMA API is accessible.
        """
        try:
            # Try to call the API
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            
            if response.status_code == 200:
                self.is_connected = True
                self.connection_error = None
                logger.info(f"Successfully connected to OLLAMA API at {self.host}")
                
                # Check if our model exists
                data = response.json()
                models = [model["name"] for model in data.get("models", [])]
                
                if self.model_name not in models:
                    logger.warning(f"Model '{self.model_name}' not found in OLLAMA. Available models: {models}")
                    
            else:
                self.is_connected = False
                self.connection_error = f"API error: {response.status_code}"
                logger.error(f"Failed to connect to OLLAMA API: {self.connection_error}")
                
        except requests.exceptions.RequestException as e:
            self.is_connected = False
            self.connection_error = str(e)
            logger.error(f"OLLAMA API connection error: {str(e)}")
            
    def wait_for_connection(self, timeout: float = 10.0) -> bool:
        """
        Wait for connection to be established.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if connection was established, False otherwise
        """
        if self.is_connected:
            return True
            
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_connected:
                return True
            time.sleep(0.5)
            
        # Timeout occurred
        return False
        
    def complete(self, prompt: str, **kwargs) -> str:
        """
        Complete a prompt using the OLLAMA API.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            str: Generated completion
        """
        # Check connection
        if not self.is_connected and not self.wait_for_connection(timeout=5.0):
            logger.error("Cannot complete prompt: OLLAMA API not connected")
            raise ConnectionError(f"OLLAMA API not connected: {self.connection_error}")
            
        try:
            # Prepare parameters
            params = self.default_params.copy()
            params.update(kwargs)
            params["prompt"] = prompt
            
            # Make API call
            url = f"{self.host}/api/generate"
            response = requests.post(url, json=params, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "")
            else:
                logger.error(f"OLLAMA API error: {response.status_code} - {response.text}")
                raise Exception(f"OLLAMA API error: {response.status_code}")
                
        except requests.exceptions.Timeout:
            logger.error("OLLAMA API request timed out")
            raise TimeoutError("OLLAMA API request timed out")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"OLLAMA API request error: {str(e)}")
            self.is_connected = False
            self.connection_error = str(e)
            raise
            
        except Exception as e:
            logger.error(f"Error in OLLAMA completion: {str(e)}")
            raise
            
    def complete_streaming(self, prompt: str, callback=None, **kwargs) -> str:
        """
        Complete a prompt with streaming response.
        
        Args:
            prompt: Input prompt
            callback: Function to call for each chunk of the response
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            str: Complete generated text
        """
        # Check connection
        if not self.is_connected and not self.wait_for_connection(timeout=5.0):
            logger.error("Cannot complete prompt: OLLAMA API not connected")
            raise ConnectionError(f"OLLAMA API not connected: {self.connection_error}")
            
        try:
            # Prepare parameters
            params = self.default_params.copy()
            params.update(kwargs)
            params["prompt"] = prompt
            params["stream"] = True
            
            # Make API call
            url = f"{self.host}/api/generate"
            response = requests.post(url, json=params, timeout=self.timeout, stream=True)
            
            if response.status_code == 200:
                full_response = ""
                
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            chunk = data.get("response", "")
                            full_response += chunk
                            
                            # Call callback if provided
                            if callback is not None:
                                callback(chunk, data)
                                
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to decode JSON from OLLAMA: {line}")
                            
                return full_response
            else:
                logger.error(f"OLLAMA API error: {response.status_code} - {response.text}")
                raise Exception(f"OLLAMA API error: {response.status_code}")
                
        except requests.exceptions.Timeout:
            logger.error("OLLAMA API request timed out")
            raise TimeoutError("OLLAMA API request timed out")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"OLLAMA API request error: {str(e)}")
            self.is_connected = False
            self.connection_error = str(e)
            raise
            
        except Exception as e:
            logger.error(f"Error in OLLAMA streaming completion: {str(e)}")
            raise
            
    def is_model_available(self, model_name: str = None) -> bool:
        """
        Check if a model is available in OLLAMA.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            bool: True if the model is available, False otherwise
        """
        if model_name is None:
            model_name = self.model_name
            
        if not self.is_connected and not self.wait_for_connection(timeout=5.0):
            logger.error("Cannot check model: OLLAMA API not connected")
            return False
            
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                models = [model["name"] for model in data.get("models", [])]
                return model_name in models
            else:
                logger.error(f"OLLAMA API error when checking models: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error checking model availability: {str(e)}")
            return False
            
    def get_available_models(self) -> List[str]:
        """
        Get a list of available models in OLLAMA.
        
        Returns:
            list: List of available model names
        """
        if not self.is_connected and not self.wait_for_connection(timeout=5.0):
            logger.error("Cannot get models: OLLAMA API not connected")
            return []
            
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            else:
                logger.error(f"OLLAMA API error when getting models: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            return []
            
    def set_model(self, model_name: str):
        """
        Set the model to use for completions.
        
        Args:
            model_name: Name of the model to use
        """
        self.model_name = model_name
        self.default_params["model"] = model_name
        logger.info(f"Set OLLAMA model to {model_name}")
        
    def get_model_status(self) -> Dict:
        """
        Get the status of the current model.
        
        Returns:
            dict: Status information
        """
        return {
            "connected": self.is_connected,
            "model": self.model_name,
            "error": self.connection_error,
            "available": self.is_model_available() if self.is_connected else False
        }