"""
WebSocket client for transmitting misalignment data.
"""
import json
import time
import threading
import queue
import websocket
from typing import Dict, List, Optional, Tuple, Any, Callable
from loguru import logger

from config import config


class WebSocketClient:
    """
    WebSocket client for transmitting misalignment data to a server.
    """
    def __init__(self, server_url: str = None):
        """
        Initialize the WebSocket client.
        
        Args:
            server_url: WebSocket server URL
        """
        self.server_url = server_url or config.websocket.server_url
        self.reconnect_interval = config.websocket.reconnect_interval
        self.max_retries = config.websocket.max_retries
        self.ping_interval = config.websocket.ping_interval
        
        # WebSocket connection
        self.ws = None
        self.is_connected = False
        self.connection_error = None
        self.connect_retry_count = 0
        
        # Message queue
        self.message_queue = queue.Queue()
        self.is_sending = False
        self.sending_thread = None
        
        # Status callback
        self.status_callback = None
        
        # Connection thread
        self.connection_thread = None
        self.is_connecting = False
        
        # Ping thread
        self.ping_thread = None
        self.is_pinging = False
        
    def set_status_callback(self, callback: Callable[[Dict], None]):
        """
        Set a callback to be called when connection status changes.
        
        Args:
            callback: Function to call with status updates
        """
        self.status_callback = callback
        
    def _notify_status(self, status: Dict):
        """
        Notify status callback of updates.
        
        Args:
            status: Status information
        """
        if self.status_callback:
            try:
                self.status_callback(status)
            except Exception as e:
                logger.error(f"Error in WebSocket status callback: {str(e)}")
                
    def connect(self, async_connect: bool = True) -> bool:
        """
        Connect to the WebSocket server.
        
        Args:
            async_connect: Whether to connect asynchronously
            
        Returns:
            bool: True if connection was initiated, False otherwise
        """
        if self.is_connected:
            logger.info("WebSocket already connected")
            return True
            
        if self.is_connecting:
            logger.info("WebSocket connection already in progress")
            return True
            
        # Connect asynchronously
        if async_connect:
            self.is_connecting = True
            self.connection_thread = threading.Thread(target=self._connect_thread, daemon=True)
            self.connection_thread.start()
            return True
            
        # Connect synchronously
        return self._connect()
        
    def _connect_thread(self):
        """
        Thread for asynchronous connection.
        """
        try:
            success = self._connect()
            
            if success:
                # Start sending thread
                if not self.is_sending:
                    self.is_sending = True
                    self.sending_thread = threading.Thread(target=self._sending_loop, daemon=True)
                    self.sending_thread.start()
                    
                # Start ping thread
                if not self.is_pinging:
                    self.is_pinging = True
                    self.ping_thread = threading.Thread(target=self._ping_loop, daemon=True)
                    self.ping_thread.start()
            
        except Exception as e:
            logger.error(f"Error in WebSocket connection thread: {str(e)}")
            self.is_connected = False
            self.connection_error = str(e)
            self._notify_status({
                "connected": False,
                "error": str(e)
            })
            
        finally:
            self.is_connecting = False
            
    def _connect(self) -> bool:
        """
        Connect to the WebSocket server.
        
        Returns:
            bool: True if connection was successful, False otherwise
        """
        try:
            logger.info(f"Connecting to WebSocket server at {self.server_url}")
            
            # Set up callbacks
            def on_open(ws):
                logger.info("WebSocket connection opened")
                self.is_connected = True
                self.connection_error = None
                self.connect_retry_count = 0
                self._notify_status({
                    "connected": True,
                    "error": None
                })
                
            def on_message(ws, message):
                logger.debug(f"WebSocket message received: {message}")
                # Process message if needed
                
            def on_error(ws, error):
                logger.error(f"WebSocket error: {str(error)}")
                self.connection_error = str(error)
                self._notify_status({
                    "connected": self.is_connected,
                    "error": str(error)
                })
                
            def on_close(ws, close_status_code, close_msg):
                was_connected = self.is_connected
                self.is_connected = False
                
                if was_connected:
                    logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
                    self._notify_status({
                        "connected": False,
                        "error": f"Connection closed: {close_status_code} - {close_msg}"
                    })
                    
            # Create WebSocket
            self.ws = websocket.WebSocketApp(
                self.server_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            # Start WebSocket in a thread
            wst = threading.Thread(target=self.ws.run_forever, daemon=True)
            wst.start()
            
            # Wait for connection to establish or fail
            for _ in range(10):  # Wait up to 5 seconds
                if self.is_connected:
                    return True
                time.sleep(0.5)
                
            # If we get here, connection didn't establish in time
            logger.warning("WebSocket connection timeout")
            self.connection_error = "Connection timeout"
            self._notify_status({
                "connected": False,
                "error": "Connection timeout"
            })
            return False
            
        except Exception as e:
            logger.error(f"Error connecting to WebSocket: {str(e)}")
            self.connection_error = str(e)
            self._notify_status({
                "connected": False,
                "error": str(e)
            })
            return False
            
    def reconnect(self):
        """
        Attempt to reconnect to the WebSocket server.
        """
        if self.is_connected:
            return
            
        if self.is_connecting:
            return
            
        self.connect_retry_count += 1
        
        if self.connect_retry_count > self.max_retries:
            logger.error(f"Maximum reconnection attempts ({self.max_retries}) reached")
            return
            
        logger.info(f"Attempting to reconnect to WebSocket (attempt {self.connect_retry_count}/{self.max_retries})")
        self.connect()
        
    def _ping_loop(self):
        """
        Loop for sending periodic pings to keep the connection alive.
        """
        while self.is_pinging:
            try:
                if self.is_connected and self.ws:
                    # Send a ping
                    self.ws.send(json.dumps({"type": "ping", "timestamp": time.time()}))
                    logger.debug("Sent WebSocket ping")
            except Exception as e:
                logger.error(f"Error sending WebSocket ping: {str(e)}")
                
            # Sleep until next ping
            time.sleep(self.ping_interval)
            
    def _sending_loop(self):
        """
        Loop for sending queued messages.
        """
        while self.is_sending:
            try:
                # Get the next message to send
                message = self.message_queue.get(timeout=1.0)
                
                # Send the message
                self._send_message(message)
                
                # Mark task as done
                self.message_queue.task_done()
                
            except queue.Empty:
                # No messages to send, just continue
                pass
                
            except Exception as e:
                logger.error(f"Error in WebSocket sending loop: {str(e)}")
                
    def _send_message(self, message: Dict) -> bool:
        """
        Send a message to the WebSocket server.
        
        Args:
            message: Message to send
            
        Returns:
            bool: True if message was sent, False otherwise
        """
        if not self.is_connected:
            logger.warning("Cannot send message: WebSocket not connected")
            # Try to reconnect
            self.reconnect()
            return False
            
        try:
            # Convert to JSON
            json_message = json.dumps(message)
            
            # Send message
            self.ws.send(json_message)
            logger.debug(f"Sent WebSocket message: {json_message[:100]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {str(e)}")
            return False
            
    def send(self, message: Dict, wait_for_connection: bool = False) -> bool:
        """
        Send a message to the WebSocket server.
        
        Args:
            message: Message to send
            wait_for_connection: Whether to wait for a connection if not connected
            
        Returns:
            bool: True if message was queued, False otherwise
        """
        # If we're not connected and not waiting for connection, try to connect
        if not self.is_connected and not self.is_connecting and not wait_for_connection:
            self.connect()
            
        # Start sending thread if not already running
        if not self.is_sending:
            self.is_sending = True
            self.sending_thread = threading.Thread(target=self._sending_loop, daemon=True)
            self.sending_thread.start()
            
        # Add message to queue
        try:
            self.message_queue.put(message)
            return True
        except Exception as e:
            logger.error(f"Error queuing WebSocket message: {str(e)}")
            return False
            
    def send_misalignment_data(self, misalignment_data: Dict) -> bool:
        """
        Send misalignment detection data to the server.
        
        Args:
            misalignment_data: Misalignment data to send
            
        Returns:
            bool: True if data was queued, False otherwise
        """
        # Wrap data in a standardized message format
        message = {
            "type": "misalignment_data",
            "timestamp": time.time(),
            "data": misalignment_data
        }
        
        return self.send(message)
        
    def close(self):
        """
        Close the WebSocket connection.
        """
        # Stop threads
        self.is_sending = False
        self.is_pinging = False
        
        if self.sending_thread is not None and self.sending_thread.is_alive():
            self.sending_thread.join(timeout=1.0)
            
        if self.ping_thread is not None and self.ping_thread.is_alive():
            self.ping_thread.join(timeout=1.0)
            
        # Close WebSocket
        if self.ws:
            self.ws.close()
            self.ws = None
            
        self.is_connected = False
        logger.info("Closed WebSocket connection")
        
    def get_status(self) -> Dict:
        """
        Get the current status of the WebSocket connection.
        
        Returns:
            dict: Status information
        """
        return {
            "connected": self.is_connected,
            "connecting": self.is_connecting,
            "error": self.connection_error,
            "server_url": self.server_url,
            "retry_count": self.connect_retry_count,
            "queue_size": self.message_queue.qsize()
        }