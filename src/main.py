"""
Main entry point for the misalignment detection system.
"""
import os
import sys
from pathlib import Path

# Add the current directory to the path so we can import modules
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from src.data import setup_logging
from src.ui import run_app


def main():
    """
    Main entry point for the application.
    """
    # Set up logging
    setup_logging()
    
    # Run the Streamlit app
    run_app()


if __name__ == "__main__":
    main()