"""
Visualization components for the Tkinter UI.
"""
import os
import time
import math
import colorsys
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")  # Must be before importing pyplot
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import cv2

from utils.config import Config
from utils.error_handler import get_logger, log_exception

logger = get_logger(__name__)

class VideoPanel(ttk.Frame):
    """Panel for displaying video feed."""
    
    def __init__(self, parent, title="Video Feed"):
        """
        Initialize the video panel.
        
        Args:
            parent: Parent widget
            title (str): Panel title
        """
        super().__init__(parent, padding=5)
        self.parent = parent
        
        # Panel title
        title_label = ttk.Label(self, text=title, font=("Arial", 12, "bold"))
        title_label.pack(pady=(0, 5))
        
        # Create frame for canvas to have more control
        self.canvas_frame = ttk.Frame(self, borderwidth=2, relief="groove")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas_frame.pack_propagate(False)  # Prevent automatic resizing
        
        # Create canvas for video display
        self.canvas = tk.Canvas(
            self.canvas_frame,
            width=Config.CAMERA_WIDTH // 2,
            height=Config.CAMERA_HEIGHT // 2,
            bg="black"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Status label
        self.status_var = tk.StringVar(value="No Video")
        self.status_label = ttk.Label(self, textvariable=self.status_var)
        self.status_label.pack(pady=(5, 0))
        
        # Initialize variables
        self.current_frame = None
        self.photo_image = None
        
        # Bind resize event
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.last_resize_time = 0
        self.current_width = Config.CAMERA_WIDTH // 2
        self.current_height = Config.CAMERA_HEIGHT // 2
    
    def _on_canvas_resize(self, event):
        """Handle canvas resize event."""
        # Limit how often we process resize events to avoid performance issues
        current_time = time.time()
        if current_time - self.last_resize_time < 0.1:  # Only process every 100ms
            return
        
        self.last_resize_time = current_time
        self.current_width = event.width
        self.current_height = event.height
        
        # If we have a current frame, update it to fit the new size
        if self.current_frame is not None:
            self._resize_and_display_frame()
    
    def _resize_and_display_frame(self):
        """Resize the current OpenCV frame and display it."""
        if not hasattr(self, 'current_cv_frame') or self.current_cv_frame is None:
            return
        
        # Get current frame
        frame = self.current_cv_frame
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Calculate scaling to maintain aspect ratio
        scale = min(self.current_width / width, self.current_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize frame
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        # Convert to RGB (from BGR)
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        # Convert to ImageTk format
        self.current_frame = Image.fromarray(rgb_frame)
        self.photo_image = ImageTk.PhotoImage(self.current_frame)
        
        # Clear the canvas before drawing
        self.canvas.delete("all")
        
        # Draw image centered in canvas
        self.canvas.create_image(
            self.current_width // 2, 
            self.current_height // 2,
            image=self.photo_image,
            anchor=tk.CENTER
        )
    
    def update_frame(self, frame):
        """
        Update the video frame.
        
        Args:
            frame: OpenCV frame (numpy array)
        """
        if frame is None:
            self.status_var.set("No Video")
            return
        
        try:
            # Store the original frame
            self.current_cv_frame = frame.copy()
            
            # Resize and display
            self._resize_and_display_frame()
            
            self.status_var.set("Video Feed Active")
        
        except Exception as e:
            log_exception(logger, e, "Error updating video frame")
            self.status_var.set(f"Error: {str(e)}")


class ScoreChart(ttk.Frame):
    """Chart for displaying misalignment scores."""
    
    def __init__(self, parent, title="Misalignment Scores"):
        """
        Initialize the score chart.
        
        Args:
            parent: Parent widget
            title (str): Chart title
        """
        super().__init__(parent, padding=5)
        self.parent = parent
        
        # Panel title
        title_label = ttk.Label(self, text=title, font=("Arial", 12, "bold"))
        title_label.pack(pady=(0, 5))
        
        # Create Matplotlib figure
        self.figure = Figure(figsize=(6, 3), dpi=100)
        self.plot = self.figure.add_subplot(111)
        self.plot.set_ylim(0, 1)
        self.plot.set_title("Misalignment Score History")
        self.plot.set_xlabel("Time")
        self.plot.set_ylabel("Score")
        
        # Frame for holding the chart
        self.chart_frame = ttk.Frame(self, borderwidth=2, relief="groove")
        self.chart_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas for the figure
        self.canvas = FigureCanvasTkAgg(self.figure, self.chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize data
        self.history_size = Config.UI_CHART_HISTORY
        self.timestamps = []
        self.scores = {}
        self.lines = {}
        self.colors = {}
    
    def _get_participant_color(self, participant_id):
        """
        Get a unique color for a participant.
        
        Args:
            participant_id (str): Participant ID
            
        Returns:
            tuple: RGB color
        """
        if participant_id in self.colors:
            return self.colors[participant_id]
        
        # Generate a unique hue based on participant ID
        hue = hash(participant_id) % 100 / 100.0
        
        # Convert HSV to RGB (using full saturation and value)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        
        # Store color for future use
        self.colors[participant_id] = (r, g, b)
        
        return r, g, b
    
    def update_scores(self, scores, timestamp=None):
        """
        Update the score chart.
        
        Args:
            scores (dict): Participant scores
            timestamp (float, optional): Timestamp for scores
        """
        if not scores:
            return
        
        try:
            if timestamp is None:
                timestamp = time.time()
            
            # Add timestamp
            self.timestamps.append(timestamp)
            
            # Add scores for each participant
            for participant_id, score in scores.items():
                if participant_id not in self.scores:
                    self.scores[participant_id] = []
                
                self.scores[participant_id].append(score)
                
                # Trim scores if necessary
                if len(self.scores[participant_id]) > self.history_size:
                    self.scores[participant_id] = self.scores[participant_id][-self.history_size:]
            
            # Trim timestamps if necessary
            if len(self.timestamps) > self.history_size:
                self.timestamps = self.timestamps[-self.history_size:]
            
            # Update plot
            self.plot.clear()
            self.plot.set_ylim(0, 1)
            self.plot.set_title("Misalignment Score History")
            self.plot.set_xlabel("Time")
            self.plot.set_ylabel("Score")
            
            # Add score lines for each participant
            for participant_id, participant_scores in self.scores.items():
                # Match scores length with timestamps length
                scores_to_plot = participant_scores[-len(self.timestamps):]
                
                # Get color for participant
                color = self._get_participant_color(participant_id)
                
                # Plot scores
                line, = self.plot.plot(
                    range(len(scores_to_plot)),
                    scores_to_plot,
                    label=participant_id,
                    color=color
                )
                
                # Store line for future updates
                self.lines[participant_id] = line
            
            # Add threshold line
            self.plot.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
            
            # Add legend
            self.plot.legend(loc='upper left')
            
            # Redraw canvas
            self.canvas.draw()
        
        except Exception as e:
            log_exception(logger, e, "Error updating score chart")


class TranscriptPanel(ttk.Frame):
    """Panel for displaying transcripts with highlighted issues."""
    
    def __init__(self, parent, title="Conversation Transcript"):
        """
        Initialize the transcript panel.
        
        Args:
            parent: Parent widget
            title (str): Panel title
        """
        super().__init__(parent, padding=5)
        self.parent = parent
        
        # Panel title
        title_label = ttk.Label(self, text=title, font=("Arial", 12, "bold"))
        title_label.pack(pady=(0, 5))
        
        # Create frame for text
        self.text_frame = ttk.Frame(self, borderwidth=2, relief="groove")
        self.text_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create text widget
        self.text = tk.Text(
            self.text_frame,
            wrap=tk.WORD,
            height=10,
            width=50,
            font=("Arial", 10)
        )
        self.text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.text_frame, command=self.text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.config(yscrollcommand=scrollbar.set)
        
        # Configure text tags
        self.text.tag_config("highlight", background="yellow")
        self.text.tag_config("timestamp", foreground="blue", font=("Arial", 8))
        self.text.tag_config("speaker", foreground="green", font=("Arial", 10, "bold"))
        
        # Initialize transcript history
        self.transcript_history = []
        self.max_history = 10
    
    def update_transcript(self, transcript, timestamp=None, highlight_words=None):
        """
        Update the transcript.
        
        Args:
            transcript (str): New transcript text
            timestamp (float, optional): Timestamp for transcript
            highlight_words (list, optional): Words to highlight
        """
        if not transcript:
            return
        
        try:
            if timestamp is None:
                timestamp = time.time()
            
            # Format timestamp
            timestamp_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
            
            # Add to history
            self.transcript_history.append({
                "text": transcript,
                "timestamp": timestamp_str,
                "highlight_words": highlight_words or []
            })
            
            # Trim history if necessary
            if len(self.transcript_history) > self.max_history:
                self.transcript_history = self.transcript_history[-self.max_history:]
            
            # Clear text widget
            self.text.delete("1.0", tk.END)
            
            # Add transcript history
            for i, entry in enumerate(self.transcript_history):
                # Add timestamp
                self.text.insert(tk.END, f"[{entry['timestamp']}] ", "timestamp")
                
                # Add transcript text
                text = entry["text"]
                
                # Highlight words if provided
                if entry["highlight_words"]:
                    # Split text into words
                    words = text.split()
                    for word in words:
                        # Check if word matches any highlight word
                        should_highlight = any(
                            highlight.lower() in word.lower()
                            for highlight in entry["highlight_words"]
                        )
                        
                        if should_highlight:
                            self.text.insert(tk.END, word, "highlight")
                        else:
                            self.text.insert(tk.END, word)
                        
                        self.text.insert(tk.END, " ")
                else:
                    self.text.insert(tk.END, text)
                
                # Add newline between entries
                if i < len(self.transcript_history) - 1:
                    self.text.insert(tk.END, "\n\n")
            
            # Scroll to end
            self.text.see(tk.END)
        
        except Exception as e:
            log_exception(logger, e, "Error updating transcript")


class AnalysisPanel(ttk.Frame):
    """Panel for displaying LLM analysis."""
    
    def __init__(self, parent, title="Misalignment Analysis"):
        """
        Initialize the analysis panel.
        
        Args:
            parent: Parent widget
            title (str): Panel title
        """
        super().__init__(parent, padding=5)
        self.parent = parent
        
        # Panel title
        title_label = ttk.Label(self, text=title, font=("Arial", 12, "bold"))
        title_label.pack(pady=(0, 5))
        
        # Create frame for results
        results_frame = ttk.LabelFrame(self, text="Results", padding=5)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Misalignment status
        self.status_var = tk.StringVar(value="No misalignment detected")
        self.status_label = ttk.Label(
            results_frame,
            textvariable=self.status_var,
            font=("Arial", 11, "bold")
        )
        self.status_label.pack(fill=tk.X, pady=5)
        
        # Cause frame
        cause_frame = ttk.LabelFrame(results_frame, text="Cause", padding=5)
        cause_frame.pack(fill=tk.BOTH, expand=True)
        
        # Cause text
        self.cause_text = tk.Text(
            cause_frame,
            wrap=tk.WORD,
            height=3,
            width=40,
            font=("Arial", 10)
        )
        self.cause_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        cause_scrollbar = ttk.Scrollbar(cause_frame, command=self.cause_text.yview)
        cause_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.cause_text.config(yscrollcommand=cause_scrollbar.set)
        
        # Recommendation frame
        recommendation_frame = ttk.LabelFrame(results_frame, text="Recommendation", padding=5)
        recommendation_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Recommendation text
        self.recommendation_text = tk.Text(
            recommendation_frame,
            wrap=tk.WORD,
            height=3,
            width=40,
            font=("Arial", 10)
        )
        self.recommendation_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        recommendation_scrollbar = ttk.Scrollbar(recommendation_frame, command=self.recommendation_text.yview)
        recommendation_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.recommendation_text.config(yscrollcommand=recommendation_scrollbar.set)
        
        # Initialize colors
        self._configure_colors()
        
        # Add a separator to provide visual indication of status
        self.separator = ttk.Separator(self, orient="horizontal")
        self.separator.pack(fill=tk.X, pady=5)
    
    def _configure_colors(self):
        """Configure colors for status based on misalignment severity."""
        # Define colors for different states
        self.colors = {
            "none": "#4CAF50",   # Green
            "low": "#FFC107",    # Yellow
            "medium": "#FF9800", # Orange
            "high": "#F44336"    # Red
        }
    
    def update_analysis(self, analysis):
        """
        Update the analysis panel.
        
        Args:
            analysis (dict): LLM analysis results
        """
        if not analysis:
            return
        
        try:
            # Update misalignment status
            misalignment_detected = analysis.get("misalignment_detected", False)
            
            # Get average score
            scores = analysis.get("scores", {})
            avg_score = sum(scores.values()) / len(scores) if scores else 0
            
            # Determine severity
            if not misalignment_detected or avg_score < 0.3:
                severity = "none"
                status_text = "No misalignment detected"
            elif avg_score < 0.5:
                severity = "low"
                status_text = "Low misalignment detected"
            elif avg_score < 0.7:
                severity = "medium"
                status_text = "Medium misalignment detected"
            else:
                severity = "high"
                status_text = "High misalignment detected"
            
            # Update status text
            self.status_var.set(status_text)
            
            # Use foreground color instead of background
            color = self.colors.get(severity, self.colors["none"])
            
            # We'll change the color of the status label instead of the whole frame
            self.status_label.configure(foreground=color)
            
            # Update cause text
            self.cause_text.delete("1.0", tk.END)
            cause = analysis.get("cause", "No cause identified")
            self.cause_text.insert("1.0", cause)
            
            # Update recommendation text
            self.recommendation_text.delete("1.0", tk.END)
            recommendation = analysis.get("recommendation", "No recommendation available")
            self.recommendation_text.insert("1.0", recommendation)
        
        except Exception as e:
            log_exception(logger, e, "Error updating analysis panel")


class ParticipantScorePanel(ttk.Frame):
    """Panel for displaying individual participant scores."""
    
    def __init__(self, parent, participant_id, title=None):
        """
        Initialize the participant score panel.
        
        Args:
            parent: Parent widget
            participant_id (str): Participant ID
            title (str, optional): Panel title
        """
        super().__init__(parent, padding=5)
        self.parent = parent
        self.participant_id = participant_id
        
        # Panel title
        if title is None:
            title = f"Participant: {participant_id}"
        
        title_label = ttk.Label(self, text=title, font=("Arial", 11, "bold"))
        title_label.pack(pady=(0, 5))
        
        # Frame for gauge
        self.gauge_frame = ttk.Frame(self, borderwidth=2, relief="groove")
        self.gauge_frame.pack(pady=5)
        
        # Create score gauge
        self.canvas = tk.Canvas(
            self.gauge_frame,
            width=150,
            height=150,
            bg="white"
        )
        self.canvas.pack(pady=5)
        
        # Score label
        self.score_var = tk.StringVar(value="0.00")
        score_label = ttk.Label(
            self,
            textvariable=self.score_var,
            font=("Arial", 16, "bold")
        )
        score_label.pack(pady=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Normal")
        self.status_label = ttk.Label(
            self,
            textvariable=self.status_var,
            font=("Arial", 10)
        )
        self.status_label.pack(pady=5)
        
        # Initialize gauge
        self.current_score = 0.0
        self._draw_gauge(self.current_score)
        
        # Create a separator to provide visual indication of status
        self.separator = ttk.Separator(self, orient="horizontal")
        self.separator.pack(fill=tk.X, pady=5)
    
    def _draw_gauge(self, score):
        """
        Draw the score gauge.
        
        Args:
            score (float): Score value (0-1)
        """
        # Clear canvas
        self.canvas.delete("all")
        
        # Get canvas dimensions
        width = self.canvas.winfo_width() or 150
        height = self.canvas.winfo_height() or 150
        
        # Calculate gauge parameters
        center_x = width // 2
        center_y = height // 2
        radius = min(width, height) // 2 - 10
        
        # Draw gauge background
        self.canvas.create_arc(
            center_x - radius, center_y - radius,
            center_x + radius, center_y + radius,
            start=135, extent=270,
            style=tk.ARC, width=20,
            outline="#EEEEEE"
        )
        
        # Determine gauge color based on score
        if score < 0.3:
            color = "#4CAF50"  # Green
        elif score < 0.5:
            color = "#FFC107"  # Yellow
        elif score < 0.7:
            color = "#FF9800"  # Orange
        else:
            color = "#F44336"  # Red
        
        # Calculate extent angle (0-270 degrees)
        extent = 270 * score
        
        # Draw score arc
        self.canvas.create_arc(
            center_x - radius, center_y - radius,
            center_x + radius, center_y + radius,
            start=135, extent=extent,
            style=tk.ARC, width=20,
            outline=color
        )
        
        # Draw center circle
        self.canvas.create_oval(
            center_x - 20, center_y - 20,
            center_x + 20, center_y + 20,
            fill=color, outline=""
        )
        
        # Draw score markers
        for i in range(5):
            angle_deg = 135 + i * 67.5
            angle_rad = math.radians(angle_deg)
            
            # Calculate marker positions
            x1 = center_x + (radius - 25) * math.cos(angle_rad)
            y1 = center_y + (radius - 25) * math.sin(angle_rad)
            x2 = center_x + (radius - 5) * math.cos(angle_rad)
            y2 = center_y + (radius - 5) * math.sin(angle_rad)
            
            # Draw marker line
            self.canvas.create_line(x1, y1, x2, y2, width=2, fill="#AAAAAA")
            
            # Draw marker text
            text_x = center_x + (radius + 15) * math.cos(angle_rad)
            text_y = center_y + (radius + 15) * math.sin(angle_rad)
            
            marker_value = i * 0.25
            self.canvas.create_text(
                text_x, text_y,
                text=f"{marker_value:.2f}",
                font=("Arial", 8)
            )
    
    def update_score(self, score):
        """
        Update the participant score.
        
        Args:
            score (float): Score value (0-1)
        """
        if score is None:
            return
        
        try:
            # Normalize score
            score = max(0.0, min(1.0, float(score)))
            
            # Update score display
            self.current_score = score
            self.score_var.set(f"{score:.2f}")
            
            # Update gauge
            self._draw_gauge(score)
            
            # Update status
            if score < 0.3:
                status = "Normal"
                self.status_label.configure(foreground="#4CAF50")  # Green
            elif score < 0.5:
                status = "Slight Misalignment"
                self.status_label.configure(foreground="#FFC107")  # Yellow
            elif score < 0.7:
                status = "Moderate Misalignment"
                self.status_label.configure(foreground="#FF9800")  # Orange
            else:
                status = "Significant Misalignment"
                self.status_label.configure(foreground="#F44336")  # Red
            
            self.status_var.set(status)
        
        except Exception as e:
            log_exception(logger, e, f"Error updating score for {self.participant_id}")