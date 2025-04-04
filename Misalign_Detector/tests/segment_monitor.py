"""
Simple WebSocket client to monitor segments from the misalignment detector.
"""
import asyncio
import json
import websockets
import argparse
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SegmentMonitor")

class SegmentMonitor:
    """Client for monitoring segments from the misalignment detector."""
    
    def __init__(self, host="localhost", port=8765):
        """
        Initialize the segment monitor.
        
        Args:
            host (str): WebSocket server host
            port (int): WebSocket server port
        """
        self.host = host
        self.port = port
        self.uri = f"ws://{host}:{port}"
        self.websocket = None
        self.segments = []
    
    async def connect(self):
        """Connect to the WebSocket server."""
        try:
            self.websocket = await websockets.connect(self.uri)
            logger.info(f"Connected to {self.uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.uri}: {str(e)}")
            return False
    
    async def listen(self):
        """Listen for segments from the server."""
        if not self.websocket:
            logger.error("Not connected")
            return
        
        try:
            logger.info("Listening for segments...")
            
            while True:
                # Wait for message
                message = await self.websocket.recv()
                
                # Parse message
                data = json.loads(message)
                
                # Check if it's a segment message
                if data.get("type") == "segment_complete":
                    self._handle_segment(data)
                else:
                    logger.info(f"Received other message: {data.get('type', 'unknown')}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed")
        except Exception as e:
            logger.error(f"Error listening for segments: {str(e)}")
    
    def _handle_segment(self, data):
        """
        Handle a segment message.
        
        Args:
            data (dict): Segment data
        """
        # Extract segment info
        segment_id = data.get("segment_id", "unknown")
        timestamp = data.get("timestamp", time.time())
        formatted_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        misalignment_detected = data.get("misalignment_detected", False)
        
        # Extract scores
        scores = data.get("scores", {})
        avg_score = sum(scores.values()) / len(scores) if scores else 0
        
        # Extract cause and recommendation
        cause = data.get("cause", "Not available")
        recommendation = data.get("recommendation", "Not available")
        
        # Print segment info
        print("\n" + "=" * 50)
        print(f"SEGMENT: {segment_id} ({formatted_time})")
        print("=" * 50)
        print(f"Misalignment Detected: {'YES' if misalignment_detected else 'NO'}")
        print(f"Average Score: {avg_score:.2f}")
        print("\nScores:")
        for participant, score in scores.items():
            print(f"  {participant}: {score:.2f}")
        
        print("\nCause:")
        print(f"  {cause}")
        
        print("\nRecommendation:")
        print(f"  {recommendation}")
        print("=" * 50)
        
        # Store segment
        self.segments.append(data)
        
        logger.info(f"Processed segment {segment_id}")
    
    async def close(self):
        """Close the WebSocket connection."""
        if self.websocket:
            await self.websocket.close()
            logger.info("Connection closed")


async def main(host, port):
    """Run the segment monitor."""
    monitor = SegmentMonitor(host, port)
    
    # Connect to server
    connected = await monitor.connect()
    if not connected:
        return
    
    # Listen for segments
    try:
        await monitor.listen()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Close connection
        await monitor.close()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Segment Monitor for Misalignment Detection System")
    parser.add_argument("--host", default="localhost", help="WebSocket server host")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket server port")
    
    args = parser.parse_args()
    
    # Run client
    try:
        asyncio.run(main(args.host, args.port))
    except KeyboardInterrupt:
        print("\nMonitor stopped")