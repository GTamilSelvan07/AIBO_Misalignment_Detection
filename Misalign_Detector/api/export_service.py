"""
Export service for session data.
"""
import os
import json
import shutil
import zipfile
from datetime import datetime

from utils.config import Config
from utils.error_handler import get_logger, log_exception
from utils.helpers import format_timestamp

logger = get_logger(__name__)

class ExportService:
    """Service for exporting session data."""
    
    def __init__(self, session_dir, data_logger):
        """
        Initialize the export service.
        
        Args:
            session_dir (str): Directory to save session data
            data_logger: Data logger instance
        """
        self.session_dir = session_dir
        self.data_logger = data_logger
        self.exports_dir = os.path.join(Config.DATA_DIR, "exports")
        
        # Ensure exports directory exists
        os.makedirs(self.exports_dir, exist_ok=True)
        
        logger.info("Initialized Export Service")
    
    def export_session(self, format="json"):
        """
        Export the current session.
        
        Args:
            format (str): Export format ("json" or "zip")
            
        Returns:
            str: Path to export file
        """
        try:
            # Generate export file path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = os.path.basename(self.session_dir)
            export_name = f"{session_id}_{timestamp}"
            
            if format == "json":
                # Export as JSON
                return self._export_json(export_name)
            elif format == "zip":
                # Export as ZIP
                return self._export_zip(export_name)
            else:
                logger.error(f"Unsupported export format: {format}")
                return None
        
        except Exception as e:
            error_msg = log_exception(logger, e, "Error exporting session")
            return None
    
    def _export_json(self, export_name):
        """
        Export session as JSON.
        
        Args:
            export_name (str): Export file name (without extension)
            
        Returns:
            str: Path to export file
        """
        try:
            # Use data logger to generate export
            export_path = self.data_logger.export_session()
            
            # Copy export to exports directory
            dest_path = os.path.join(self.exports_dir, f"{export_name}.json")
            shutil.copy2(export_path, dest_path)
            
            logger.info(f"Exported session to JSON: {dest_path}")
            return dest_path
        
        except Exception as e:
            log_exception(logger, e, "Error exporting session as JSON")
            return None
    
    def _export_zip(self, export_name):
        """
        Export session as ZIP.
        
        Args:
            export_name (str): Export file name (without extension)
            
        Returns:
            str: Path to export file
        """
        try:
            # Generate ZIP file path
            zip_path = os.path.join(self.exports_dir, f"{export_name}.zip")
            
            # Create ZIP file
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # First, add the JSON export
                export_path = self.data_logger.export_session()
                zipf.write(export_path, os.path.basename(export_path))
                
                # Add all CSV files
                for root, _, files in os.walk(self.session_dir):
                    for file in files:
                        if file.endswith('.csv'):
                            file_path = os.path.join(root, file)
                            # Get relative path within session directory
                            rel_path = os.path.relpath(file_path, self.session_dir)
                            zipf.write(file_path, rel_path)
                
                # Add all transcripts
                transcript_dir = os.path.join(self.session_dir, "transcripts")
                if os.path.exists(transcript_dir):
                    for file in os.listdir(transcript_dir):
                        if file.endswith('.json'):
                            file_path = os.path.join(transcript_dir, file)
                            rel_path = os.path.join("transcripts", file)
                            zipf.write(file_path, rel_path)
                
                # Add analysis files
                analysis_dir = os.path.join(self.session_dir, "analysis")
                if os.path.exists(analysis_dir):
                    for file in os.listdir(analysis_dir):
                        if file.endswith('.json'):
                            file_path = os.path.join(analysis_dir, file)
                            rel_path = os.path.join("analysis", file)
                            zipf.write(file_path, rel_path)
            
            logger.info(f"Exported session to ZIP: {zip_path}")
            return zip_path
        
        except Exception as e:
            log_exception(logger, e, "Error exporting session as ZIP")
            return None
    
    def list_exports(self):
        """
        List all available exports.
        
        Returns:
            list: List of export file paths
        """
        try:
            exports = []
            
            for file in os.listdir(self.exports_dir):
                if file.endswith(('.json', '.zip')):
                    file_path = os.path.join(self.exports_dir, file)
                    
                    # Get file info
                    stat = os.stat(file_path)
                    file_size = stat.st_size
                    modified_time = format_timestamp(stat.st_mtime)
                    
                    exports.append({
                        "path": file_path,
                        "name": file,
                        "size": file_size,
                        "modified": modified_time
                    })
            
            return sorted(exports, key=lambda x: x["modified"], reverse=True)
        
        except Exception as e:
            log_exception(logger, e, "Error listing exports")
            return []