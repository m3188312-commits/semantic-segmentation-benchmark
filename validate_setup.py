#!/usr/bin/env python3
"""
Validation script for Semantic Segmentation Benchmark setup
"""
import sys
from pathlib import Path
import importlib

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    required_packages = {
        "torch": "PyTorch for deep learning",
        "ultralytics": "YOLO training framework (CRITICAL)",
        "numpy": "Numerical computing",
        "cv2": "OpenCV for image processing",
        "yaml": "YAML configuration parsing",
        "tqdm": "Progress bars"
    }
    
    optional_packages = {
        "torchvision": "PyTorch vision utilities (optional for YOLO)"
    }
    
    all_installed = True
    
    # Check required packages
    for package, description in required_packages.items():
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            print(f"   {description}")
            all_installed = False
    
    # Check optional packages
    print("\nOptional packages:")
    for package, description in optional_packages.items():
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"⚠️  {package} - NOT INSTALLED (optional)")
            print(f"   {description}")
    
    if not all_installed:
        print(f"\n❌ Missing required packages: {', '.join([pkg for pkg, desc in required_packages.items() if not importlib.util.find_spec(pkg)])}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    return True

def check_dataset_structure():
    """Check if dataset structure is correct."""
    dataset_dir = Path("dataset")
    
    if not dataset_dir.exists():
        print("❌ Dataset directory not found")
        return False
    
    required_splits = ["train", "test", "lowres"]
    required_subdirs = ["image", "mask", "labels"]
    
    for split in required_splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            print(f"❌ Missing split directory: {split}")
            continue
            
        print(f"\n📁 Checking {split} split:")
        for subdir in required_subdirs:
            subdir_path = split_dir / subdir
            if subdir_path.exists():
                files = list(subdir_path.glob("*"))
                print(f"   ✅ {subdir}: {len(files)} files")
            else:
                print(f"   ❌ {subdir}: Missing")
                return False
    
    return True

def check_yolo_config():
    """Check YOLO configuration files."""
    yolo_dir = Path("models/yolo")
    
    if not yolo_dir.exists():
        print("❌ YOLO directory not found")
        return False
    
    config_files = ["train.yaml", "lowres.yaml", "classes.yaml"]
    
    print("\n🔧 Checking YOLO configuration:")
    for config in config_files:
        config_path = yolo_dir / config
        if config_path.exists():
            print(f"   ✅ {config}")
        else:
            print(f"   ❌ {config}: Missing")
            return False
    
    return True

def check_labels_exist():
    """Check if label files have been generated."""
    dataset_dir = Path("dataset")
    
    print("\n🏷️ Checking label files:")
    for split in ["train", "test", "lowres"]:
        labels_dir = dataset_dir / split / "labels"
        if labels_dir.exists():
            label_files = list(labels_dir.glob("*.txt"))
            print(f"   ✅ {split}: {len(label_files)} label files")
            
            if len(label_files) == 0:
                print(f"   ⚠️ {split}: No label files found. Run dataset.py first!")
                return False
        else:
            print(f"   ❌ {split}: Labels directory missing")
            return False
    
    return True

def main():
    """Run all validation checks."""
    print("🔍 Validating Semantic Segmentation Benchmark Setup\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Dataset Structure", check_dataset_structure),
        ("YOLO Configuration", check_yolo_config),
        ("Label Files", check_labels_exist)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n{'='*50}")
        print(f"Checking: {check_name}")
        print('='*50)
        
        try:
            if not check_func():
                all_passed = False
        except Exception as e:
            print(f"❌ Error during {check_name} check: {e}")
            all_passed = False
    
    print(f"\n{'='*50}")
    if all_passed:
        print("🎉 All checks passed! Your setup is ready for training.")
        print("\nNext steps:")
        print("1. cd models/yolo")
        print("2. python train.py")
    else:
        print("❌ Some checks failed. Please fix the issues above before training.")
        print("\nCommon fixes:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Generate labels: cd models/yolo && python dataset.py")
        print("3. Check dataset structure matches README requirements")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
