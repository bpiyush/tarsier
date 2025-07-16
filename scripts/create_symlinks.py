#!/usr/bin/env python3
"""Create symlinks for .MP4 files with lowercase extensions."""

import os
import glob
from pathlib import Path
from tqdm import tqdm

def create_symlinks():
    # Base directories
    base_dir = "/scratch/shared/beegfs/piyush/datasets/EPIC-Kitchens-100"
    cut_clips_dir = os.path.join(base_dir, "cut_clips")
    symlinks_dir = os.path.join(base_dir, "symlinks_to_cut_clips")
    
    # Find all .MP4 files
    print("Finding all .MP4 files...")
    mp4_files = glob.glob(os.path.join(cut_clips_dir, "**/*.MP4"), recursive=True)
    print(f"Found {len(mp4_files)} .MP4 files")
    
    # Create symlinks with progress bar
    print("Creating symlinks...")
    for mp4_file in tqdm(mp4_files, desc="Creating symlinks"):
        # Get relative path from cut_clips directory
        rel_path = os.path.relpath(mp4_file, cut_clips_dir)
        
        # Get directory path for symlink
        rel_dir = os.path.dirname(rel_path)
        symlink_dir = os.path.join(symlinks_dir, rel_dir)
        
        # Get filename with lowercase extension
        filename = os.path.basename(mp4_file)
        lowercase_filename = filename.replace('.MP4', '.mp4')
        
        # Create directory if it doesn't exist
        os.makedirs(symlink_dir, exist_ok=True)
        
        # Create symlink with absolute path
        symlink_path = os.path.join(symlink_dir, lowercase_filename)
        if not os.path.exists(symlink_path):
            os.symlink(mp4_file, symlink_path)

if __name__ == "__main__":
    create_symlinks()
    print("✅ Symlink creation complete!") 