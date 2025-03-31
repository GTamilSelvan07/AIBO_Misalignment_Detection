"""
Script to run the Streamlit app for the misalignment detection system.
"""
import os
import sys
from pathlib import Path

# Add the current directory to the path so we can import modules
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))


from src.data import setup_logging



if __name__ == "__main__":
    # Set up logging
    setup_logging()
    
    # Run the Streamlit app using `streamlit run`
    try:
        import streamlit.web.cli as stcli
        
        # Get the full path to the app module
        app_path = os.path.join(current_dir, "src", "ui", "app.py")
        
        # Run the Streamlit app
        sys.argv = ["streamlit", "run", app_path, "--server.headless", "true"]
        stcli.main()
        
    except Exception as e:
        print(f"Error running Streamlit app: {str(e)}")
        print("You can run the app manually with: streamlit run src/ui/app.py")
        sys.exit(1)