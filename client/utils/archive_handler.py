import os
import zipfile
import time
import shutil
from datetime import datetime

def create_backup_on_exit():
    """Create a backup archive of result files when the program exits, and delete the original files after successful backup"""
    try:
        print("Creating backup archive...")
        
        # Create archive directory (if it doesn't exist)
        archive_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "archive")
        if not os.path.exists(archive_dir):
            os.makedirs(archive_dir)
            
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = os.path.join(archive_dir, f"results_backup_{timestamp}.zip")
        
        # Files and directories to backup
        json_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "result_image.json")
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
        
        # Check if files exist
        files_to_backup = []
        if os.path.exists(json_file):
            files_to_backup.append(("result_image.json", json_file))
        
        if os.path.exists(results_dir):
            files_to_backup.append(("results", results_dir))
            
        if not files_to_backup:
            print("No files or directories found for backup")
            return
        
        # Create ZIP file
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add JSON file
            for name, path in files_to_backup:
                if os.path.isfile(path):
                    # If it's a file, add it directly
                    zipf.write(path, os.path.basename(path))
                    print(f"Added file: {os.path.basename(path)}")
                else:
                    # If it's a directory, add all files inside it
                    for root, dirs, files in os.walk(path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            # Calculate relative path in the zip
                            relative_path = os.path.join("results", os.path.relpath(file_path, start=results_dir))
                            zipf.write(file_path, relative_path)
                            print(f"Added file: {relative_path}")
        
        print(f"Backup complete! File saved to: {zip_filename}")
        
        # Verify backup file was created successfully
        if os.path.exists(zip_filename) and os.path.getsize(zip_filename) > 0:
            # Delete original files
            delete_original_files(json_file, results_dir)
            return zip_filename
        else:
            print("Warning: Backup file was not created successfully, original files will not be deleted")
            return None
        
    except Exception as e:
        print(f"Error creating backup: {str(e)}")
        return None

def delete_original_files(json_file, results_dir):
    """Delete the original result_image.json and all files in the results directory"""
    try:
        print("Starting cleanup of original files...")
        
        # Delete result_image.json
        if os.path.exists(json_file):
            os.remove(json_file)
            print(f"Deleted: {json_file}")
        
        # Delete all files in the results directory, but keep the directory structure
        if os.path.exists(results_dir):
            # Traverse and delete files
            for root, dirs, files in os.walk(results_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    except Exception as e:
                        print(f"Failed to delete file {file_path}: {e}")
            
            print(f"All files in results directory have been cleared")
        
        print("Original file cleanup complete")
    
    except Exception as e:
        print(f"Error deleting original files: {e}")