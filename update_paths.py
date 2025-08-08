#!/usr/bin/env python3
"""
Update file paths in the codebase after folder renaming
Changes data_processing/ -> data_processing/ and streamlit_app/ -> streamlit_app/
"""

import os
import glob

def update_file_paths():
    """Update all file paths in the codebase"""
    
    # Path mappings
    path_mappings = {
        "data_processing/": "data_processing/",
        "streamlit_app/": "streamlit_app/",
        "datasets/": "datasets/",
        "from data_processing.": "from data_processing.",
        "import data_processing.": "import data_processing.",
        "from streamlit_app.": "from streamlit_app.",
        "import streamlit_app.": "import streamlit_app.",
    }
    
    # File extensions to process
    file_extensions = ["*.py", "*.txt", "*.md", "*.json"]
    
    print("🔄 Starting path updates...")
    
    # Get all files to process
    files_to_process = []
    for ext in file_extensions:
        files_to_process.extend(glob.glob(ext, recursive=True))
        files_to_process.extend(glob.glob(f"**/{ext}", recursive=True))
    
    # Remove duplicates and exclude certain files
    files_to_process = list(set(files_to_process))
    files_to_process = [f for f in files_to_process if not f.startswith('.git')]
    
    updated_files = []
    
    for file_path in files_to_process:
        if os.path.isfile(file_path):
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Apply path mappings
                for old_path, new_path in path_mappings.items():
                    content = content.replace(old_path, new_path)
                
                # Write back if content changed
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    updated_files.append(file_path)
                    print(f"✅ Updated: {file_path}")
                    
            except Exception as e:
                print(f"❌ Failed to update {file_path}: {e}")
    
    print(f"\n🎉 Path updates completed! Updated {len(updated_files)} files.")
    
    if updated_files:
        print("\n📋 Updated files:")
        for file_path in updated_files:
            print(f"  - {file_path}")

if __name__ == "__main__":
    update_file_paths() 