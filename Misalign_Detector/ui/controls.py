"""
Control components for the Tkinter UI.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time

from utils.config import Config
from utils.error_handler import get_logger, log_exception

logger = get_logger(__name__)

class ControlPanel(ttk.Frame):
    """Panel for system controls (start/stop, manual controls)."""
    
    def __init__(self, parent, on_start=None, on_stop=None, on_export=None, on_send=None, on_manual_analysis=None):
        """
        Initialize the control panel.
        
        Args:
            parent: Parent widget
            on_start (callable, optional): Callback for start button
            on_stop (callable, optional): Callback for stop button
            on_export (callable, optional): Callback for export button
            on_send (callable, optional): Callback for send button
            on_manual_analysis (callable, optional): Callback for manual analysis
        """
        super().__init__(parent, padding=10)
        self.parent = parent
        
        # Store callbacks
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_export = on_export
        self.on_send = on_send
        self.on_manual_analysis = on_manual_analysis
        
        # Create control buttons frame
        buttons_frame = ttk.Frame(self)
        buttons_frame.pack(fill=tk.X, pady=5)
        
        # Recording control frame
        recording_frame = ttk.LabelFrame(self, text="Recording Controls", padding=5)
        recording_frame.pack(fill=tk.X, pady=5)
        
        # Start button
        self.start_button = ttk.Button(
            recording_frame,
            text="▶ Start Recording",
            command=self._on_start_click,
            style="Accent.TButton"
        )
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Send button (analyze and save current segment)
        self.send_button = ttk.Button(
            recording_frame,
            text="📤 Send & Save",
            command=self._on_send_click,
            state=tk.DISABLED
        )
        self.send_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Stop button
        self.stop_button = ttk.Button(
            recording_frame,
            text="⏹ Stop Recording",
            command=self._on_stop_click,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Export button
        self.export_button = ttk.Button(
            recording_frame,
            text="💾 Export Session",
            command=self._on_export_click,
            state=tk.DISABLED
        )
        self.export_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(
            self,
            textvariable=self.status_var,
            font=("Arial", 10, "italic")
        )
        status_label.pack(fill=tk.X, pady=5)
        
        # Create frame for manual analysis
        manual_frame = ttk.LabelFrame(self, text="Manual Analysis", padding=5)
        manual_frame.pack(fill=tk.X, pady=5, expand=True)
        
        # Manual analysis text input
        self.manual_text = tk.Text(
            manual_frame,
            wrap=tk.WORD,
            height=4,
            width=50,
            font=("Arial", 10)
        )
        self.manual_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Add scrollbar
        manual_scrollbar = ttk.Scrollbar(manual_frame, command=self.manual_text.yview)
        manual_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.manual_text.config(yscrollcommand=manual_scrollbar.set)
        
        # Analysis button
        self.analyze_button = ttk.Button(
            manual_frame,
            text="🔍 Analyze",
            command=self._on_analyze_click,
            state=tk.DISABLED
        )
        self.analyze_button.pack(side=tk.BOTTOM, pady=5)
        
        # Placeholder text
        self.manual_text.insert("1.0", "Enter context for manual analysis here...")
        self.manual_text.bind("<FocusIn>", self._clear_placeholder)
        self.manual_text.bind("<FocusOut>", self._add_placeholder)
        
        # Current recording time
        self.recording_time_var = tk.StringVar(value="00:00:00")
        self.recording_time_label = ttk.Label(
            self,
            text="Recording Time:",
            font=("Arial", 10)
        )
        self.recording_time_label.pack(side=tk.LEFT, padx=(0,5))
        
        self.recording_time_display = ttk.Label(
            self,
            textvariable=self.recording_time_var,
            font=("Arial", 11, "bold")
        )
        self.recording_time_display.pack(side=tk.LEFT)
        
        # Running state
        self.is_running = False
        self.is_recording = False
        self.recording_start_time = None
        self.recording_timer_id = None
    
    def _clear_placeholder(self, event):
        """Clear placeholder text when focused."""
        if self.manual_text.get("1.0", "end-1c") == "Enter context for manual analysis here...":
            self.manual_text.delete("1.0", tk.END)
    
    def _add_placeholder(self, event):
        """Add placeholder text when unfocused and empty."""
        if not self.manual_text.get("1.0", "end-1c").strip():
            self.manual_text.delete("1.0", tk.END)
            self.manual_text.insert("1.0", "Enter context for manual analysis here...")
    
    def _on_start_click(self):
        """Handle start button click."""
        if self.on_start:
            try:
                # Call start callback
                success = self.on_start()
                
                if success:
                    # Update UI state
                    self.is_running = True
                    self.is_recording = True
                    self.start_button.config(state=tk.DISABLED)
                    self.send_button.config(state=tk.NORMAL)
                    self.stop_button.config(state=tk.NORMAL)
                    self.analyze_button.config(state=tk.NORMAL)
                    self.status_var.set("Recording in progress...")
                    
                    # Start recording timer
                    self.recording_start_time = time.time()
                    self._update_recording_time()
            
            except Exception as e:
                log_exception(logger, e, "Error starting recording")
                messagebox.showerror("Error", f"Failed to start recording: {str(e)}")
    
    def _on_send_click(self):
        """Handle send button click (analyze and save current segment)."""
        if self.on_send:
            try:
                # Call send callback
                success = self.on_send()
                
                if success:
                    # Update UI state - keep system running but start new recording segment
                    self.is_recording = True
                    self.status_var.set("Segment saved and sent! Starting new segment...")
                    
                    # Reset recording timer
                    self.recording_start_time = time.time()
                    self._update_recording_time()
            
            except Exception as e:
                log_exception(logger, e, "Error sending/saving segment")
                messagebox.showerror("Error", f"Failed to process segment: {str(e)}")
    
    def _on_stop_click(self):
        """Handle stop button click."""
        if self.on_stop:
            try:
                # Call stop callback
                success = self.on_stop()
                
                if success:
                    # Update UI state
                    self.is_running = False
                    self.is_recording = False
                    self.start_button.config(state=tk.NORMAL)
                    self.send_button.config(state=tk.DISABLED)
                    self.stop_button.config(state=tk.DISABLED)
                    self.export_button.config(state=tk.NORMAL)
                    self.analyze_button.config(state=tk.DISABLED)
                    self.status_var.set("Recording stopped and saved")
                    
                    # Stop recording timer
                    if self.recording_timer_id:
                        self.after_cancel(self.recording_timer_id)
                        self.recording_timer_id = None
            
            except Exception as e:
                log_exception(logger, e, "Error stopping recording")
                messagebox.showerror("Error", f"Failed to stop recording: {str(e)}")
    
    def _on_export_click(self):
        """Handle export button click."""
        if self.on_export:
            try:
                # Create file dialog
                formats = [
                    ("JSON Files", "*.json"),
                    ("ZIP Archives", "*.zip"),
                    ("All Files", "*.*")
                ]
                
                export_format = messagebox.askquestion(
                    "Export Format",
                    "Do you want to export as ZIP (includes all data)?\n\n"
                    "Select 'Yes' for ZIP, 'No' for JSON."
                )
                
                format_type = "zip" if export_format == "yes" else "json"
                
                # Call export callback
                self.status_var.set("Exporting session...")
                
                # Use threading to avoid freezing UI
                def export_thread():
                    try:
                        export_path = self.on_export(format_type)
                        
                        if export_path:
                            self.status_var.set(f"Exported to {export_path}")
                            messagebox.showinfo("Export Complete", f"Session exported to:\n{export_path}")
                        else:
                            self.status_var.set("Export failed")
                            messagebox.showerror("Export Failed", "Failed to export session.")
                    
                    except Exception as e:
                        log_exception(logger, e, "Error exporting session")
                        self.status_var.set("Export error")
                        messagebox.showerror("Export Error", f"Error exporting session: {str(e)}")
                
                threading.Thread(target=export_thread, daemon=True).start()
            
            except Exception as e:
                log_exception(logger, e, "Error exporting session")
                messagebox.showerror("Export Error", f"Error exporting session: {str(e)}")
    
    def _on_analyze_click(self):
        """Handle analyze button click."""
        if self.on_manual_analysis:
            try:
                # Get text from input
                context = self.manual_text.get("1.0", tk.END).strip()
                
                if context == "Enter context for manual analysis here...":
                    messagebox.showinfo("Input Required", "Please enter context for analysis.")
                    return
                
                if not context:
                    messagebox.showinfo("Input Required", "Please enter context for analysis.")
                    return
                
                # Call analysis callback
                self.status_var.set("Analyzing...")
                
                # Use threading to avoid freezing UI
                def analysis_thread():
                    try:
                        result = self.on_manual_analysis(context)
                        
                        if result:
                            self.status_var.set("Analysis complete")
                        else:
                            self.status_var.set("Analysis failed")
                            messagebox.showerror("Analysis Failed", "Failed to analyze context.")
                    
                    except Exception as e:
                        log_exception(logger, e, "Error in manual analysis")
                        self.status_var.set("Analysis error")
                        messagebox.showerror("Analysis Error", f"Error analyzing context: {str(e)}")
                
                threading.Thread(target=analysis_thread, daemon=True).start()
            
            except Exception as e:
                log_exception(logger, e, "Error in manual analysis")
                messagebox.showerror("Analysis Error", f"Error analyzing context: {str(e)}")
    
    def _update_recording_time(self):
        """Update the recording time display."""
        if self.is_recording and self.recording_start_time:
            # Calculate elapsed time
            elapsed = time.time() - self.recording_start_time
            
            # Format as HH:MM:SS
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            
            self.recording_time_var.set(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
            
            # Update every second
            self.recording_timer_id = self.after(1000, self._update_recording_time)
    
    def set_running_state(self, is_running):
        """
        Set the running state of the control panel.
        
        Args:
            is_running (bool): Whether the system is running
        """
        self.is_running = is_running
        
        if is_running:
            self.start_button.config(state=tk.DISABLED)
            self.send_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.NORMAL)
            self.analyze_button.config(state=tk.NORMAL)
            self.status_var.set("Recording in progress...")
            
            # Start recording timer
            self.is_recording = True
            self.recording_start_time = time.time()
            self._update_recording_time()
        else:
            self.start_button.config(state=tk.NORMAL)
            self.send_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.DISABLED)
            self.export_button.config(state=tk.NORMAL)
            self.analyze_button.config(state=tk.DISABLED)
            self.status_var.set("Recording stopped")
            
            # Stop recording timer
            self.is_recording = False
            if self.recording_timer_id:
                self.after_cancel(self.recording_timer_id)
                self.recording_timer_id = None


class SettingsPanel(ttk.Frame):
    """Panel for system settings."""
    
    def __init__(self, parent, on_settings_change=None):
        """
        Initialize the settings panel.
        
        Args:
            parent: Parent widget
            on_settings_change (callable, optional): Callback for settings changes
        """
        super().__init__(parent, padding=10)
        self.parent = parent
        self.on_settings_change = on_settings_change
        
        # Create settings notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create camera settings tab
        camera_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(camera_frame, text="Camera")
        
        # Camera device selection
        camera_label = ttk.Label(
            camera_frame,
            text="Camera Devices:",
            font=("Arial", 10, "bold")
        )
        camera_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # Participant 1 camera
        ttk.Label(camera_frame, text="Participant 1:").grid(row=1, column=0, sticky=tk.W)
        self.camera1_var = tk.IntVar(value=Config.CAMERA_DEVICE_IDS[0] if len(Config.CAMERA_DEVICE_IDS) > 0 else 0)
        camera1_spinner = ttk.Spinbox(
            camera_frame,
            from_=0,
            to=10,
            textvariable=self.camera1_var,
            width=5
        )
        camera1_spinner.grid(row=1, column=1, sticky=tk.W)
        
        # Participant 2 camera
        ttk.Label(camera_frame, text="Participant 2:").grid(row=2, column=0, sticky=tk.W)
        self.camera2_var = tk.IntVar(value=Config.CAMERA_DEVICE_IDS[1] if len(Config.CAMERA_DEVICE_IDS) > 1 else 1)
        camera2_spinner = ttk.Spinbox(
            camera_frame,
            from_=0,
            to=10,
            textvariable=self.camera2_var,
            width=5
        )
        camera2_spinner.grid(row=2, column=1, sticky=tk.W)
        
        # Camera resolution
        ttk.Label(camera_frame, text="Resolution:").grid(row=3, column=0, sticky=tk.W, pady=(10, 0))
        
        # Width
        ttk.Label(camera_frame, text="Width:").grid(row=4, column=0, sticky=tk.W)
        self.width_var = tk.IntVar(value=Config.CAMERA_WIDTH)
        width_spinner = ttk.Spinbox(
            camera_frame,
            from_=320,
            to=1920,
            increment=160,
            textvariable=self.width_var,
            width=5
        )
        width_spinner.grid(row=4, column=1, sticky=tk.W)
        
        # Height
        ttk.Label(camera_frame, text="Height:").grid(row=5, column=0, sticky=tk.W)
        self.height_var = tk.IntVar(value=Config.CAMERA_HEIGHT)
        height_spinner = ttk.Spinbox(
            camera_frame,
            from_=240,
            to=1080,
            increment=120,
            textvariable=self.height_var,
            width=5
        )
        height_spinner.grid(row=5, column=1, sticky=tk.W)
        
        # Create audio settings tab
        audio_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(audio_frame, text="Audio")
        
        # Audio device selection
        audio_label = ttk.Label(
            audio_frame,
            text="Audio Settings:",
            font=("Arial", 10, "bold")
        )
        audio_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # Sample rate
        ttk.Label(audio_frame, text="Sample Rate:").grid(row=1, column=0, sticky=tk.W)
        self.sample_rate_var = tk.IntVar(value=Config.AUDIO_SAMPLE_RATE)
        sample_rate_combo = ttk.Combobox(
            audio_frame,
            textvariable=self.sample_rate_var,
            values=[8000, 16000, 22050, 44100, 48000],
            width=10
        )
        sample_rate_combo.grid(row=1, column=1, sticky=tk.W)
        
        # Analysis settings tab
        analysis_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(analysis_frame, text="Analysis")
        
        # Analysis settings
        analysis_label = ttk.Label(
            analysis_frame,
            text="Analysis Settings:",
            font=("Arial", 10, "bold")
        )
        analysis_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # Analysis interval
        ttk.Label(analysis_frame, text="Interval (ms):").grid(row=1, column=0, sticky=tk.W)
        self.analysis_interval_var = tk.IntVar(value=Config.ANALYSIS_INTERVAL_MS)
        interval_spinner = ttk.Spinbox(
            analysis_frame,
            from_=100,
            to=2000,
            increment=100,
            textvariable=self.analysis_interval_var,
            width=5
        )
        interval_spinner.grid(row=1, column=1, sticky=tk.W)
        
        # Detector weights
        ttk.Label(analysis_frame, text="Detector Weights:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        
        # Facial weight
        ttk.Label(analysis_frame, text="Facial:").grid(row=3, column=0, sticky=tk.W)
        self.facial_weight_var = tk.DoubleVar(value=0.4)
        facial_scale = ttk.Scale(
            analysis_frame,
            from_=0.0,
            to=1.0,
            variable=self.facial_weight_var,
            orient=tk.HORIZONTAL,
            length=100
        )
        facial_scale.grid(row=3, column=1, sticky=tk.W)
        
        # LLM weight
        ttk.Label(analysis_frame, text="LLM:").grid(row=4, column=0, sticky=tk.W)
        self.llm_weight_var = tk.DoubleVar(value=0.6)
        llm_scale = ttk.Scale(
            analysis_frame,
            from_=0.0,
            to=1.0,
            variable=self.llm_weight_var,
            orient=tk.HORIZONTAL,
            length=100
        )
        llm_scale.grid(row=4, column=1, sticky=tk.W)
        
        # Create save button
        save_button = ttk.Button(
            self,
            text="Save Settings",
            command=self._on_save_settings
        )
        save_button.pack(pady=10)
    
    def _on_save_settings(self):
        """Handle save settings button click."""
        try:
            # Collect settings
            settings = {
                "camera1_device": self.camera1_var.get(),
                "camera2_device": self.camera2_var.get(),
                "camera_width": self.width_var.get(),
                "camera_height": self.height_var.get(),
                "audio_sample_rate": self.sample_rate_var.get(),
                "analysis_interval_ms": self.analysis_interval_var.get(),
                "facial_weight": self.facial_weight_var.get(),
                "llm_weight": self.llm_weight_var.get()
            }
            
            # Update Config
            Config.CAMERA_DEVICE_IDS = [settings["camera1_device"], settings["camera2_device"]]
            Config.CAMERA_WIDTH = settings["camera_width"]
            Config.CAMERA_HEIGHT = settings["camera_height"]
            Config.AUDIO_SAMPLE_RATE = settings["audio_sample_rate"]
            Config.ANALYSIS_INTERVAL_MS = settings["analysis_interval_ms"]
            
            # Call callback if provided
            if self.on_settings_change:
                self.on_settings_change(settings)
            
            # Show success message
            messagebox.showinfo("Settings Saved", "Settings have been saved successfully.")
        
        except Exception as e:
            log_exception(logger, e, "Error saving settings")
            messagebox.showerror("Settings Error", f"Error saving settings: {str(e)}")
    
    def get_settings(self):
        """
        Get current settings.
        
        Returns:
            dict: Current settings
        """
        return {
            "camera1_device": self.camera1_var.get(),
            "camera2_device": self.camera2_var.get(),
            "camera_width": self.width_var.get(),
            "camera_height": self.height_var.get(),
            "audio_sample_rate": self.sample_rate_var.get(),
            "analysis_interval_ms": self.analysis_interval_var.get(),
            "facial_weight": self.facial_weight_var.get(),
            "llm_weight": self.llm_weight_var.get()
        }


class StatusBar(ttk.Frame):
    """Status bar for displaying system status."""
    
    def __init__(self, parent):
        """
        Initialize the status bar.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.parent = parent
        
        # Create status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(
            self,
            textvariable=self.status_var,
            padding=(5, 2)
        )
        status_label.pack(side=tk.LEFT)
        
        # Create progress bar
        self.progress = ttk.Progressbar(
            self,
            mode="indeterminate",
            length=100
        )
        self.progress.pack(side=tk.RIGHT, padx=5)
        
        # Time label
        self.time_var = tk.StringVar(value="00:00:00")
        time_label = ttk.Label(
            self,
            textvariable=self.time_var,
            padding=(5, 2)
        )
        time_label.pack(side=tk.RIGHT)
        
        # Start time update thread
        self.time_thread = threading.Thread(target=self._update_time, daemon=True)
        self.time_thread.start()
    
    def _update_time(self):
        """Update the time display."""
        while True:
            try:
                current_time = time.strftime("%H:%M:%S")
                self.time_var.set(current_time)
                time.sleep(1)
            except Exception:
                # Ignore errors in time thread
                pass
    
    def set_status(self, status, show_progress=False):
        """
        Set the status text.
        
        Args:
            status (str): Status text
            show_progress (bool): Whether to show the progress bar
        """
        self.status_var.set(status)
        
        if show_progress:
            self.progress.start(10)
        else:
            self.progress.stop()
    
    def clear(self):
        """Clear the status bar."""
        self.status_var.set("")
        self.progress.stop()