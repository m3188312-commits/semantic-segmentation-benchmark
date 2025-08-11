#!/usr/bin/env python3
"""
Fix YOLO dataset structure by copying labels to image directories.
This allows YOLO to find both images and labels in the same location.
"""

import shutil
from pathlib import Path
import os

def fix_yolo_structure():
    """Copy label files to image directories for YOLO compatibility."""
    
    # Get paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    dataset_dir = project_root / "dataset"
    
    print("🔧 Fixing YOLO dataset structure...")
    
    # Process each dataset split
    splits = ["train", "test", "lowres"]
    
    for split in splits:
        image_dir = dataset_dir / split / "image"
        labels_dir = dataset_dir / split / "labels"
        
        if not image_dir.exists():
            print(f"⚠️  Image directory not found: {image_dir}")
            continue
            
        if not labels_dir.exists():
            print(f"⚠️  Labels directory not found: {labels_dir}")
            continue
        
        print(f"\n📁 Processing {split} split...")
        
        # Get all label files
        label_files = list(labels_dir.glob("*.txt"))
        print(f"   Found {len(label_files)} label files")
        
        # Copy each label file to the image directory
        copied_count = 0
        for label_file in label_files:
            dest_path = image_dir / label_file.name
            
            # Check if destination already exists
            if dest_path.exists():
                print(f"   ⚠️  Label already exists: {label_file.name}")
                continue
                
            try:
                shutil.copy2(label_file, dest_path)
                copied_count += 1
            except Exception as e:
                print(f"   ❌ Error copying {label_file.name}: {e}")
        
        print(f"   ✅ Copied {copied_count} label files to {image_dir}")
    
    print("\n🎯 YOLO dataset structure fixed!")
    print("   - Labels are now in image directories")
    print("   - YOLO can find both images and labels")
    print("   - Training should work without 'no labels found' errors")

if __name__ == "__main__":
    fix_yolo_structure()
