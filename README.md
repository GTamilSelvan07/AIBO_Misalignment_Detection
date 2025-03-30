# AIBO_Misalignment_Detection
 
Prototype system that detects misalignment and misunderstanding during conversations using both camera and speech inputs. The system will:

Capture facial expressions via camera every 5 seconds to detect confusion using OpenFace features
Convert speech to text and analyze it for signs of misalignment or misunderstanding using an OLLAMA-based LLM
Generate misalignment scores based on both inputs
Provide a Streamlit-based UI for visualizing and controlling the system
Save results to log files and generate JSON for API transmission

## Misalignment Detection System - Folder Structure

```
misalignment_detection/
├── README.md                 # Project documentation
├── requirements.txt          # Project dependencies
├── .env.example              # Example environment variables
├── main.py                   # Application entry point
├── config.py                 # Configuration settings
├── run.py                    # Streamlit app launcher
│
├── src/                      # Source code
│   ├── __init__.py
│   │
│   ├── camera/               # Camera processing module
│   │   ├── __init__.py
│   │   ├── capture.py        # Camera capture functionality
│   │   ├── face_detector.py  # Face detection using OpenFace
│   │   └── misalignment.py   # Visual misalignment scoring
│   │
│   ├── speech/               # Speech processing module
│   │   ├── __init__.py
│   │   ├── recorder.py       # Audio recording functionality
│   │   ├── transcriber.py    # Speech-to-text conversion
│   │   └── analysis.py       # Speech misalignment analysis
│   │
│   ├── llm/                  # LLM integration
│   │   ├── __init__.py
│   │   ├── ollama_client.py  # OLLAMA API client
│   │   ├── prompts.py        # LLM prompts for misalignment detection
│   │   └── response_parser.py # Parse LLM responses
│   │
│   ├── scoring/              # Scoring logic
│   │   ├── __init__.py
│   │   ├── camera_score.py   # Camera-based scoring
│   │   ├── speech_score.py   # Speech-based scoring
│   │   └── combined_score.py # Combined scoring algorithm
│   │
│   ├── data/                 # Data handling
│   │   ├── __init__.py
│   │   ├── logger.py         # Logging functionality
│   │   ├── json_generator.py # JSON output generation
│   │   └── websocket.py      # WebSocket client for data transmission
│   │
│   └── ui/                   # Streamlit UI
│       ├── __init__.py
│       ├── app.py            # Main Streamlit application
│       ├── components.py     # UI components
│       ├── visualizations.py # Data visualization
│       └── settings.py       # User settings handling
│
├── tests/                    # Unit and integration tests
│   ├── __init__.py
│   ├── test_camera.py
│   ├── test_speech.py
│   ├── test_llm.py
│   ├── test_scoring.py
│   └── test_data.py
│
├── models/                   # Saved models and weights
│   └── openface/             # OpenFace models
│
├── logs/                     # Log files
│   ├── app.log
│   └── misalignment_scores/  # CSV log files for misalignment scores
│
└── docs/                     # Documentation
    ├── setup.md              # Setup instructions
    ├── usage.md              # Usage guide
    ├── api.md                # API documentation
    └── diagrams/             # System architecture diagrams
```

## File Purposes and Responsibilities

### Root Directory

- **README.md**: Project overview, quick start guide, and general information
- **requirements.txt**: All Python dependencies
- **.env.example**: Template for environment variables (API keys, server URLs)
- **main.py**: Core application logic and component integration
- **config.py**: Configuration parameters and settings
- **run.py**: Script to launch the Streamlit application

### Source Code (src/)

#### Camera Module (src/camera/)
- **capture.py**: Handles camera initialization, frame capture at specified intervals
- **face_detector.py**: Implements OpenFace integration for facial feature extraction
- **misalignment.py**: Algorithms to convert facial features to misalignment scores

#### Speech Module (src/speech/)
- **recorder.py**: Manages microphone access and audio recording
- **transcriber.py**: Converts audio to text using speech recognition
- **analysis.py**: Initial analysis of transcribed text before LLM processing

#### LLM Integration (src/llm/)
- **ollama_client.py**: Client for connecting to OLLAMA API
- **prompts.py**: Defines the prompts for the LLM to detect misalignment
- **response_parser.py**: Extracts relevant information from LLM responses

#### Scoring Logic (src/scoring/)
- **camera_score.py**: Algorithms for camera-based misalignment scoring
- **speech_score.py**: Algorithms for speech-based misalignment scoring
- **combined_score.py**: Methods to combine multiple scores into a unified metric

#### Data Handling (src/data/)
- **logger.py**: Handles logging of scores and system events
- **json_generator.py**: Creates JSON outputs in the required format
- **websocket.py**: Manages WebSocket connections and data transmission

#### UI Module (src/ui/)
- **app.py**: Main Streamlit application structure
- **components.py**: Reusable UI components (buttons, sliders, etc.)
- **visualizations.py**: Charts and visualizations for misalignment scores
- **settings.py**: UI for adjusting system parameters

### Tests Directory (tests/)
- Contains unit and integration tests for each module

### Models Directory (models/)
- Stores pre-trained models and weights for OpenFace

### Logs Directory (logs/)
- Stores application logs and misalignment score history

### Documentation (docs/)
- Comprehensive documentation for setup, usage, and architecture