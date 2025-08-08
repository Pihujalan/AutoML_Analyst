#!/usr/bin/env python3
"""
Cleanup script for AutoML Analyst repository
Removes sensitive files, large model files, and generated reports
"""

import os
import shutil
import glob

def cleanup_repository():
    """Clean up the repository by removing sensitive and generated files"""
    
    # Files to remove (sensitive and large)
    files_to_remove = [
        # Model files (large binary files)
        "*.pkl",
        "*.joblib",
        "*.h5",
        
        # Generated reports
        "epic_final_report.txt",
        "auto_training_report.txt",
        "business_insights_prompt.txt",
        
        # Generated data files
        "*.csv",
        "residuals.csv",
        "cleaned_input.csv",
        
        # Generated JSON files
        "*_metrics.json",
        "*_importance.json",
        "*_results.csv",
        "*_predictions.csv",
        "segment_analysis_*.json",
        "gbm_shap_importance.json",
        "rf_shap_importance.json",
        "final_model_summary.json",
        "best_rf_metrics.json",
        
        # Python cache
        "__pycache__",
        "*.pyc",
        "*.pyo",
        
        # IDE files
        ".vscode",
        ".idea",
        "*.swp",
        "*.swo",
        
        # OS files
        ".DS_Store",
        "Thumbs.db",
        "desktop.ini",
        
        # Logs
        "*.log",
        "logs",
        
        # Temporary files
        "*.tmp",
        "*.temp",
        "temp",
        "tmp",
        
        # Jupyter checkpoints
        "*.ipynb_checkpoints",
    ]
    
    # Directories to remove
    dirs_to_remove = [
        "plots",
        "segment_csvs",
        "__pycache__",
        ".vscode",
        ".idea",
        "logs",
        "temp",
        "tmp",
    ]
    
    print("🧹 Starting repository cleanup...")
    
    # Remove files
    for pattern in files_to_remove:
        if "*" in pattern:
            # Handle glob patterns
            for file_path in glob.glob(pattern, recursive=True):
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                        print(f"✅ Removed: {file_path}")
                    except Exception as e:
                        print(f"❌ Failed to remove {file_path}: {e}")
        else:
            # Handle specific files
            if os.path.isfile(pattern):
                try:
                    os.remove(pattern)
                    print(f"✅ Removed: {pattern}")
                except Exception as e:
                    print(f"❌ Failed to remove {pattern}: {e}")
    
    # Remove directories
    for dir_path in dirs_to_remove:
        if os.path.isdir(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"✅ Removed directory: {dir_path}")
            except Exception as e:
                print(f"❌ Failed to remove directory {dir_path}: {e}")
    
    print("🎉 Cleanup completed!")
    print("\n📋 Next steps:")
    print("1. Review the changes with: git status")
    print("2. Add the cleaned files: git add .")
    print("3. Commit the cleanup: git commit -m 'Clean repository for public release'")
    print("4. Push to GitHub: git push origin main")

if __name__ == "__main__":
    cleanup_repository() 