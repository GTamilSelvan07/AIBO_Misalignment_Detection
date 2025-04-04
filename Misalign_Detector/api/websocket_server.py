"""
WebSocket server for real-time communication with other applications.
"""
import asyncio
import json
import time
import threading
import websockets
from datetime import datetime
from threading import Thread, Event

from utils.config import Config
from utils.error_handler import get_logger, WebSocketError, log_exception

logger = get_logger(__name__)

class WebSocketServer:
    """WebSocket server for real-time communication."""
    
    def __init__(self, detector):
        """
        Initialize the WebSocket server.
        
        Args:
            detector: Misalignment detector instance
        """
        self.detector = detector
        self.host = Config.WEBSOCKET_HOST
        self.port = Config.WEBSOCKET_PORT
        
        # Server state
        self.is_running = False
        self.server = None
        self.server_thread = None
        self.stop_event = Event()
        
        # Store connected clients
        self.clients = set()
        
        # Update thread for sending regular updates
        self.update_thread = None
        self.update_interval = 0.5  # seconds
        
        # Store last sent data to avoid redundant updates
        self.last_sent_data = None
        
        logger.info(f"Initialized WebSocket Server on {self.host}:{self.port}")
    
    async def _handler(self, websocket, path):
        """
        Handle WebSocket connections.
        
        Args:
            websocket: WebSocket connection
            path: Connection path
        """
        try:
            # Register client
            self.clients.add(websocket)
            client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
            logger.info(f"Client connected: {client_info}")
            
            # Send initial data
            initial_data = self._prepare_data()
            await websocket.send(json.dumps(initial_data))
            
            # Handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    response = await self._handle_message(data)
                    
                    if response:
                        await websocket.send(json.dumps(response))
                
                except json.JSONDecodeError:
                    logger.warning(f"Received invalid JSON from {client_info}")
                    await websocket.send(json.dumps({
                        "error": "Invalid JSON format"
                    }))
                
                except Exception as e:
                    log_exception(logger, e, f"Error handling message from {client_info}")
                    await websocket.send(json.dumps({
                        "error": f"Error processing message: {str(e)}"
                    }))
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_info}")
        
        except Exception as e:
            log_exception(logger, e, f"WebSocket handler error for {client_info}")
        
        finally:
            # Unregister client
            self.clients.remove(websocket)
    
    async def _handle_message(self, data):
        """
        Handle incoming messages from clients.
        
        Args:
            data (dict): Message data
            
        Returns:
            dict: Response data
        """
        try:
            message_type = data.get("type")
            
            if message_type == "get_detection":
                # Send latest detection
                return self._prepare_data()
            
            elif message_type == "manual_analysis":
                # Trigger manual LLM analysis
                context = data.get("context", "")
                if context:
                    analysis = self.detector.llm_analyzer.manual_analysis(context)
                    return {
                        "type": "manual_analysis_result",
                        "analysis": analysis
                    }
                else:
                    return {
                        "type": "error",
                        "message": "Missing context for manual analysis"
                    }
            
            else:
                return {
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                }
        
        except Exception as e:
            log_exception(logger, e, "Error handling WebSocket message")
            return {
                "type": "error",
                "message": f"Error processing message: {str(e)}"
            }
    
    def _prepare_data(self):
        """
        Prepare data to send to clients.
        
        Returns:
            dict: Data to send
        """
        detection = self.detector.get_latest_detection()
        
        if not detection:
            return {
                "type": "no_data",
                "timestamp": time.time()
            }
        
        # Prepare simplified data for clients
        data = {
            "type": "detection",
            "timestamp": detection["timestamp"],
            "formatted_time": detection["formatted_time"],
            "misalignment_detected": detection.get("misalignment_detected", False),
            "scores": detection.get("combined_scores", {}),
            "transcript": detection.get("transcript", "")
        }
        
        # Add LLM analysis if available
        if "llm_analysis" in detection:
            data["cause"] = detection["llm_analysis"].get("cause", "")
            data["recommendation"] = detection["llm_analysis"].get("recommendation", "")
        
        return data
    
    async def _update_loop(self):
        """Send regular updates to all connected clients."""
        while not self.stop_event.is_set():
            try:
                # Check if there are any clients
                if not self.clients:
                    await asyncio.sleep(0.5)
                    continue
                
                # Prepare data
                data = self._prepare_data()
                
                # Check if data has changed
                data_json = json.dumps(data)
                if data_json != self.last_sent_data:
                    # Send to all clients
                    for client in self.clients.copy():
                        try:
                            await client.send(data_json)
                        except Exception as e:
                            # Client probably disconnected
                            pass
                    
                    # Update last sent data
                    self.last_sent_data = data_json
                
                await asyncio.sleep(self.update_interval)
            
            except Exception as e:
                log_exception(logger, e, "Error in WebSocket update loop")
                await asyncio.sleep(1.0)
    
    async def _start_server(self):
        """Start the WebSocket server."""
        try:
            # Start server
            self.server = await websockets.serve(
                self._handler, self.host, self.port
            )
            
            # Start update loop
            asyncio.create_task(self._update_loop())
            
            # Keep server running
            await self.stop_event.wait()
            
            # Close server
            self.server.close()
            await self.server.wait_closed()
        
        except Exception as e:
            error_msg = log_exception(logger, e, "Error starting WebSocket server")
            raise WebSocketError(error_msg)
    
    def _run_server(self):
        """Run the server in the asyncio event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._start_server())
        except Exception as e:
            log_exception(logger, e, "Error in WebSocket server thread")
        finally:
            loop.close()
    
    def start(self):
        """Start the WebSocket server."""
        if self.is_running:
            logger.warning("WebSocket Server already running")
            return False
        
        try:
            # Clear stop event
            self.stop_event.clear()
            
            # Start server thread
            self.server_thread = Thread(target=self._run_server, daemon=True)
            self.server_thread.start()
            
            self.is_running = True
            logger.info(f"Started WebSocket Server on {self.host}:{self.port}")
            return True
        
        except Exception as e:
            error_msg = log_exception(logger, e, "Failed to start WebSocket Server")
            raise WebSocketError(error_msg)
    
    def stop(self):
        """Stop the WebSocket server."""
        if not self.is_running:
            return
        
        # Set stop event
        self.stop_event.set()
        
        # Wait for server thread to finish
        if self.server_thread:
            self.server_thread.join(timeout=5.0)
        
        self.is_running = False
        logger.info("Stopped WebSocket Server")
    
    def broadcast(self, data):
        """
        Broadcast data to all connected clients.
        
        Args:
            data (dict): Data to broadcast
        """
        if not self.is_running or not self.clients:
            logger.warning("WebSocket server not running or no clients connected")
            return
        
        # Convert to JSON
        try:
            data_json = json.dumps(data)
            
            # Create task to send to all clients
            async def _broadcast():
                try:
                    for client in self.clients.copy():
                        try:
                            await client.send(data_json)
                            logger.info(f"Broadcast sent to client {client.remote_address}")
                        except Exception as e:
                            # Client probably disconnected
                            logger.warning(f"Failed to send to client {client.remote_address}: {str(e)}")
                except Exception as e:
                    log_exception(logger, e, "Error in broadcast execution")
            
            # Run in the server's event loop if it exists
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(_broadcast(), loop)
            else:
                logger.warning("Event loop not running, cannot broadcast")
        
        except Exception as e:
            log_exception(logger, e, "Error broadcasting data")