# Misalignment Detection System

A real-time system for detecting communication misalignment between two participants using multi-modal analysis of facial features, speech transcription, and LLM-powered conversation analysis.

## System Overview

The Misalignment Detection System is designed to detect, analyze, and log instances of communication misalignment between two participants in real-time. Using a combination of facial expression analysis, speech-to-text transcription, and language model inference, the system provides immediate feedback about potential miscommunication, helping moderators or researchers identify and address issues as they occur.

![System Architecture](docs/images/architecture.png)

### Key Features

- **Multi-modal Analysis**: Combines facial expressions, verbal content, and LLM analysis
- **Real-time Processing**: Low latency analysis (under 1s per inference)
- **Segment-based Recording**: Capture, analyze, and save specific interaction segments
- **Local Deployment**: All processing runs locally for privacy and data security
- **WebSocket Communication**: Real-time data streaming to external applications
- **Comprehensive Logging**: Session data saved in accessible JSON and CSV formats

## System Architecture

### Core Components

The system follows a modular architecture with the following key components:

1. **Camera Manager**: Captures video from two participant cameras and processes facial expressions
   - Uses OpenFace for facial feature extraction
   - Detects facial Action Units (AUs) associated with confusion and interest
   - Provides real-time feature extraction for each participant

2. **Audio Manager**: Records and transcribes conversation in real-time
   - Uses Whisper for fast and accurate speech transcription
   - Segments audio for efficient processing
   - Maintains transcript history and synchronized timestamps

3. **LLM Analyzer**: Interprets conversation context for misalignment detection
   - Leverages Gemma 3.1b via OLLAMA for lightweight local inference
   - Analyzes transcripts and facial feature data
   - Provides cause analysis and recommendations when misalignment is detected

4. **Misalignment Detector**: Combines multi-modal signals to detect communication issues
   - Weighted fusion of facial and linguistic signals
   - Thresholding for misalignment detection
   - Time-series analysis and segment-based processing

5. **Data Logger**: Comprehensive data management system
   - Session-based organization with unique IDs
   - CSV and JSON logging for analysis
   - Segment-based saving and export capabilities

### API Layer

1. **WebSocket Server**: Real-time data communication with external applications
   - Broadcasts detection events and session data
   - Supports real-time client monitoring
   - Provides granular segment data with misalignment analysis

2. **Export Service**: Facilitates data export for post-session analysis
   - JSON exports of analysis results
   - ZIP archives of complete session data
   - Structured format for research and analysis

### UI Components

1. **Main Application**: Tkinter-based interface for system control and visualization
   - Real-time video feeds for both participants
   - Score visualization and history charting
   - Transcript display with highlighted issues
   - Analysis results and recommendations

2. **Control Panel**: Recording and session management
   - Start/stop recording functionality
   - Segment-based recording with "Send & Save" capability
   - Manual analysis capabilities
   - Export functionality

3. **Visualization Components**: Real-time data visualization
   - Participant score gauges with color-coded status
   - Time-series charts of misalignment scores
   - Transcript panel with issue highlighting
   - Analysis panel with causes and recommendations

## Implementation Details

### Data Flow

1. **Capture**: Video and audio are captured from cameras and microphone
2. **Feature Extraction**: OpenFace processes facial expressions, Whisper transcribes speech
3. **Analysis**: LLM analyzes transcript and facial features for misalignment indicators
4. **Integration**: Detector combines signals into comprehensive misalignment scores
5. **Visualization**: UI displays analysis results, scores, and recommendations
6. **Export/Communication**: Data saved locally and broadcasted via WebSocket

### Misalignment Scoring

The system uses a weighted approach to calculate misalignment scores:

```
score = (facial_weight * facial_score) + (llm_weight * llm_score)
```

Where:
- `facial_score` is derived from facial Action Units associated with confusion
- `llm_score` is derived from the LLM's analysis of the conversation transcript
- Weights are configurable in the settings panel

### Segment-Based Recording

The system implements a segment-based recording approach:

1. Start recording session
2. Capture continuous data from all modalities
3. "Send & Save" button triggers:
   - Immediate analysis of current segment
   - Saving segment data to disk
   - WebSocket broadcast of results
   - Starting a new recording segment
4. Stop recording finalizes the session

This approach allows for focused analysis of specific conversation segments while maintaining a continuous recording capability.

## Deployment & Usage

### Prerequisites

- Python 3.8 or higher
- Windows 10/11 (for OpenFace compatibility)
- Two webcams for participant facial analysis
- Microphone for audio capture
- OLLAMA local LLM server with Gemma 3.1b model

### Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Install OpenFace 2.2.0 and update the path in `utils/config.py`
4. Install Faster-Whisper and update the path in `utils/config.py`
5. Set up OLLAMA and pull the Gemma 3.1b model: `ollama pull gemma:3b`

### Usage Guide

1. **Starting the System**:
   ```bash
   python main.py
   ```

2. **System Workflow**:
   - Start a recording session by clicking "Start Recording"
   - Monitor participant misalignment scores in real-time
   - Click "Send & Save" to process and save the current segment
   - Click "Stop Recording" when done
   - Export the session for later analysis

3. **Monitoring Segments**:
   - Run the segment monitor tool to observe WebSocket data:
   ```bash
   python tools/segment_monitor.py
   ```

## Technical Challenges & Lessons Learned

### UI Development Challenges

1. **Tkinter Widget Styling**:
   - **Challenge**: Tkinter's ttk widgets do not directly support background color changes using the `.configure(background=color)` method.
   - **Solution**: Used foreground color changes and visual separators for status indication instead.
   - **Learning**: When working with ttk, style configuration requires using the `ttk.Style()` system rather than direct widget configuration.

2. **Video Display Sizing**:
   - **Challenge**: Video frames would continuously grow in size during real-time display.
   - **Solution**: Implemented proper canvas resizing with size limiting and aspect ratio preservation.
   - **Learning**: Canvas widgets need explicit management for continuous video streaming applications.

### Real-time Processing Challenges

1. **Latency Management**:
   - **Challenge**: Achieving sub-second response time with multiple processing-intensive components.
   - **Solution**: Implemented asynchronous processing pipelines and optimized feature extraction intervals.
   - **Learning**: Careful tuning of processing intervals is critical for real-time applications.

2. **Resource Coordination**:
   - **Challenge**: Coordinating access to camera and audio resources without conflicts.
   - **Solution**: Implemented thread-safe resource management with clear ownership hierarchy.
   - **Learning**: Explicit resource management is essential in multi-threaded applications.

### Integration Challenges

1. **OpenFace Integration**:
   - **Challenge**: OpenFace is a separate executable that needed to be integrated into the Python pipeline.
   - **Solution**: Created a batch processing approach with file-based communication.
   - **Learning**: External tool integration often requires careful file management and subprocess handling.

2. **LLM Response Parsing**:
   - **Challenge**: Ensuring consistent structured output from the LLM for downstream processing.
   - **Solution**: Implemented robust parsing with fallback mechanisms for malformed responses.
   - **Learning**: LLM outputs require defensive programming approaches with well-defined error handling.

## Future Improvements

1. **Enhanced Analysis**:
   - Incorporate more sophisticated misalignment detection models
   - Add sentiment analysis for emotional context
   - Implement gaze tracking for attention analysis

2. **Usability Enhancements**:
   - Add session comparison tools
   - Implement automatic segment detection
   - Create annotation capabilities for researchers

3. **Technical Improvements**:
   - Migrate to GPU acceleration for feature extraction
   - Implement streaming transcription for lower latency
   - Add support for remote participants through network cameras

