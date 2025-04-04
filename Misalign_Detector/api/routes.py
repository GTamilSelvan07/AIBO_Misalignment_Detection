"""
Routes for potential future REST API endpoints.
"""
import os
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
from threading import Thread
import urllib.parse

from utils.config import Config
from utils.error_handler import get_logger, log_exception

logger = get_logger(__name__)

class APIRoutes:
    """Basic HTTP server for API endpoints."""
    
    def __init__(self, detector=None, export_service=None, port=8080):
        """
        Initialize the API routes.
        
        Args:
            detector: Misalignment detector instance
            export_service: Export service instance
            port (int): Server port
        """
        self.detector = detector
        self.export_service = export_service
        self.port = port
        self.is_running = False
        self.server = None
        self.server_thread = None
        
        logger.info(f"Initialized API Routes on port {self.port}")
    
    def start(self):
        """Start the HTTP server."""
        if self.is_running:
            logger.warning("API server already running")
            return False
        
        try:
            # Create request handler with access to detector and export service
            api_instance = self
            
            class RequestHandler(BaseHTTPRequestHandler):
                def do_GET(self):
                    """Handle GET requests."""
                    try:
                        # Parse URL
                        parsed_url = urllib.parse.urlparse(self.path)
                        path = parsed_url.path
                        
                        # Route requests
                        if path == "/api/status":
                            self._handle_status()
                        elif path == "/api/detection":
                            self._handle_detection()
                        elif path == "/api/exports":
                            self._handle_exports()
                        elif path.startswith("/api/export/"):
                            export_id = path.split("/")[-1]
                            self._handle_export(export_id)
                        else:
                            self._handle_not_found()
                    
                    except Exception as e:
                        log_exception(logger, e, f"Error handling request: {self.path}")
                        self._handle_error(str(e))
                
                def do_POST(self):
                    """Handle POST requests."""
                    try:
                        # Parse URL
                        parsed_url = urllib.parse.urlparse(self.path)
                        path = parsed_url.path
                        
                        # Get request body
                        content_length = int(self.headers.get('Content-Length', 0))
                        body = self.rfile.read(content_length).decode('utf-8')
                        
                        # Parse JSON
                        try:
                            data = json.loads(body) if body else {}
                        except json.JSONDecodeError:
                            self._handle_bad_request("Invalid JSON")
                            return
                        
                        # Route requests
                        if path == "/api/analyze":
                            self._handle_analyze(data)
                        elif path == "/api/export":
                            self._handle_create_export(data)
                        else:
                            self._handle_not_found()
                    
                    except Exception as e:
                        log_exception(logger, e, f"Error handling request: {self.path}")
                        self._handle_error(str(e))
                
                def _send_response_json(self, status_code, data):
                    """Send JSON response."""
                    self.send_response(status_code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps(data).encode('utf-8'))
                
                def _handle_status(self):
                    """Handle status request."""
                    if not api_instance.detector:
                        self._send_response_json(503, {"status": "unavailable", "message": "Detector not available"})
                        return
                    
                    # Check if detector is running
                    is_running = api_instance.detector.is_running
                    
                    self._send_response_json(200, {
                        "status": "running" if is_running else "stopped",
                        "timestamp": api_instance.detector.latest_detection["timestamp"] if api_instance.detector.latest_detection else None
                    })
                
                def _handle_detection(self):
                    """Handle detection request."""
                    if not api_instance.detector:
                        self._send_response_json(503, {"status": "unavailable", "message": "Detector not available"})
                        return
                    
                    # Get latest detection
                    detection = api_instance.detector.get_latest_detection()
                    
                    if not detection:
                        self._send_response_json(404, {"status": "not_found", "message": "No detection available"})
                        return
                    
                    self._send_response_json(200, detection)
                
                def _handle_exports(self):
                    """Handle exports list request."""
                    if not api_instance.export_service:
                        self._send_response_json(503, {"status": "unavailable", "message": "Export service not available"})
                        return
                    
                    # Get exports list
                    exports = api_instance.export_service.list_exports()
                    
                    self._send_response_json(200, {"exports": exports})
                
                def _handle_export(self, export_id):
                    """Handle export download request."""
                    if not api_instance.export_service:
                        self._send_response_json(503, {"status": "unavailable", "message": "Export service not available"})
                        return
                    
                    # Get exports list
                    exports = api_instance.export_service.list_exports()
                    
                    # Find export
                    export = next((e for e in exports if os.path.basename(e["path"]) == export_id), None)
                    
                    if not export:
                        self._send_response_json(404, {"status": "not_found", "message": f"Export {export_id} not found"})
                        return
                    
                    # Send file
                    try:
                        with open(export["path"], "rb") as f:
                            self.send_response(200)
                            content_type = "application/zip" if export["path"].endswith(".zip") else "application/json"
                            self.send_header("Content-Type", content_type)
                            self.send_header("Content-Disposition", f"attachment; filename=\"{os.path.basename(export['path'])}\"")
                            self.end_headers()
                            self.wfile.write(f.read())
                    except Exception as e:
                        log_exception(logger, e, f"Error sending export file: {export['path']}")
                        self._handle_error(f"Error sending export file: {str(e)}")
                
                def _handle_analyze(self, data):
                    """Handle analyze request."""
                    if not api_instance.detector or not api_instance.detector.llm_analyzer:
                        self._send_response_json(503, {"status": "unavailable", "message": "LLM analyzer not available"})
                        return
                    
                    # Get context
                    context = data.get("context", "")
                    
                    if not context:
                        self._send_response_json(400, {"status": "bad_request", "message": "Missing context parameter"})
                        return
                    
                    # Run manual analysis
                    try:
                        analysis = api_instance.detector.llm_analyzer.manual_analysis(context)
                        self._send_response_json(200, {"analysis": analysis})
                    except Exception as e:
                        log_exception(logger, e, "Error running manual analysis")
                        self._handle_error(f"Error running analysis: {str(e)}")
                
                def _handle_create_export(self, data):
                    """Handle create export request."""
                    if not api_instance.export_service:
                        self._send_response_json(503, {"status": "unavailable", "message": "Export service not available"})
                        return
                    
                    # Get format
                    format = data.get("format", "json")
                    
                    if format not in ["json", "zip"]:
                        self._send_response_json(400, {"status": "bad_request", "message": "Invalid format parameter"})
                        return
                    
                    # Create export
                    try:
                        export_path = api_instance.export_service.export_session(format=format)
                        
                        if not export_path:
                            self._send_response_json(500, {"status": "error", "message": "Failed to create export"})
                            return
                        
                        self._send_response_json(200, {
                            "status": "success",
                            "export": {
                                "path": export_path,
                                "name": os.path.basename(export_path),
                                "url": f"/api/export/{os.path.basename(export_path)}"
                            }
                        })
                    except Exception as e:
                        log_exception(logger, e, "Error creating export")
                        self._handle_error(f"Error creating export: {str(e)}")
                
                def _handle_not_found(self):
                    """Handle not found request."""
                    self._send_response_json(404, {"status": "not_found", "message": "Endpoint not found"})
                
                def _handle_bad_request(self, message):
                    """Handle bad request."""
                    self._send_response_json(400, {"status": "bad_request", "message": message})
                
                def _handle_error(self, message):
                    """Handle error."""
                    self._send_response_json(500, {"status": "error", "message": message})
                
                def log_message(self, format, *args):
                    """Override log_message to use our logger."""
                    logger.info(f"API: {args[0].split()[0]} {args[0].split()[1]} {args[1]}")
            
            # Start server
            self.server = HTTPServer(("", self.port), RequestHandler)
            self.server_thread = Thread(target=self._run_server, daemon=True)
            self.server_thread.start()
            
            self.is_running = True
            logger.info(f"Started API server on port {self.port}")
            return True
        
        except Exception as e:
            error_msg = log_exception(logger, e, "Failed to start API server")
            return False
    
    def _run_server(self):
        """Run the HTTP server."""
        try:
            self.server.serve_forever()
        except Exception as e:
            log_exception(logger, e, "Error in API server")
        finally:
            self.is_running = False
    
    def stop(self):
        """Stop the HTTP server."""
        if not self.is_running:
            return
        
        try:
            # Shutdown server
            if self.server:
                self.server.shutdown()
                self.server.server_close()
            
            # Wait for server thread to finish
            if self.server_thread:
                self.server_thread.join(timeout=5.0)
            
            self.is_running = False
            logger.info("Stopped API server")
        
        except Exception as e:
            log_exception(logger, e, "Error stopping API server")