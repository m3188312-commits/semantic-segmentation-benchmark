#!/usr/bin/env python3
"""
Evaluation script for pseudo-label experiments.
Evaluates all 18 cases: 3 image counts × 3 agreement thresholds × 2 unknown pixel handling methods.
"""

import os
import shutil
import json
import time
from pathlib import Path
import argparse
import subprocess
import sys

# Configuration
ORIGINAL_TRAIN = Path("dataset/train")
TEST_SET = Path("dataset/test")
PSEUDO_LABELS = Path("data/pseudo_labels")
PSEUDO_IMAGES = Path("data/pseudo_images")
OUTPUT_DIR = Path("evaluations/pseudo_label_results")
TEMP_DIR = Path("temp_evaluation")

# Evaluation cases
IMAGE_COUNTS = [10, 50, 78]  # 78 is the max we have
AGREEMENT_LEVELS = [2, 3, 4]
UNKNOWN_HANDLING = ["keep", "remove"]

def setup_directories():
    """Create necessary directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)
    
    # Create subdirectories for each case
    for img_count in IMAGE_COUNTS:
        for agreement in AGREEMENT_LEVELS:
            for handling in UNKNOWN_HANDLING:
                case_dir = OUTPUT_DIR / f"case_{img_count}_{agreement}_{handling}"
                case_dir.mkdir(exist_ok=True)

def create_training_dataset(img_count, agreement, handling):
    """Create a training dataset by combining original data with pseudo-labels."""
    case_name = f"case_{img_count}_{agreement}_{handling}"
    case_dir = TEMP_DIR / case_name
    
    # Clean up previous case
    if case_dir.exists():
        shutil.rmtree(case_dir)
    
    # Create case directory structure
    (case_dir / "image").mkdir(parents=True)
    (case_dir / "mask").mkdir(parents=True)
    
    # Copy original training data
    print(f"  Copying {len(list(ORIGINAL_TRAIN.glob('image/*.jpg')))} original training images...")
    for img_path in ORIGINAL_TRAIN.glob("image/*.jpg"):
        shutil.copy2(img_path, case_dir / "image" / img_path.name)
    
    for mask_path in ORIGINAL_TRAIN.glob("mask/*.png"):
        shutil.copy2(mask_path, case_dir / "mask" / mask_path.name)
    
    # Add pseudo-labels
    pseudo_dir = PSEUDO_LABELS / f"agreement_{agreement}"
    pseudo_img_dir = PSEUDO_IMAGES / f"agreement_{agreement}"
    
    # Get list of available pseudo-labels
    available_pseudo = list(pseudo_dir.glob("*.png"))
    
    if img_count > len(available_pseudo):
        print(f"  Warning: Requested {img_count} images but only {len(available_pseudo)} available")
        img_count = len(available_pseudo)
    
    # Select pseudo-labels (take first N)
    selected_pseudo = available_pseudo[:img_count]
    
    print(f"  Adding {len(selected_pseudo)} pseudo-labels...")
    for pseudo_path in selected_pseudo:
        stem = pseudo_path.stem
        
        if handling == "keep":
            # Use masked images (unknown pixels = (155,155,155))
            img_path = pseudo_img_dir / f"{stem}.jpg"
            if img_path.exists():
                shutil.copy2(img_path, case_dir / "image" / f"{stem}_pseudo.jpg")
            else:
                # Fallback to original unlabeled image
                orig_img = Path("dataset/unlabeled/image") / f"{stem}.jpg"
                if orig_img.exists():
                    shutil.copy2(orig_img, case_dir / "image" / f"{stem}_pseudo.jpg")
        else:
            # Use original unlabeled images
            orig_img = Path("dataset/unlabeled/image") / f"{stem}.jpg"
            if orig_img.exists():
                shutil.copy2(orig_img, case_dir / "image" / f"{stem}_pseudo.jpg")
        
        # Copy pseudo-label mask
        shutil.copy2(pseudo_path, case_dir / "mask" / f"{stem}_pseudo.png")
    
    return case_dir

def train_model(model_type, train_dir, case_name):
    """Train a model on the given training directory."""
    print(f"  Training {model_type} model...")
    
    # Model-specific training commands
    if model_type == "deeplab":
        cmd = [
            sys.executable, "models/deeplab/train.py",
            "--data_dir", str(train_dir),
            "--output_dir", str(OUTPUT_DIR / case_name / "deeplab"),
            "--epochs", "10",  # Reduced for faster evaluation
            "--batch_size", "4"
        ]
    elif model_type == "unet":
        cmd = [
            sys.executable, "models/unet_no_patches/train.py",
            "--data_dir", str(train_dir),
            "--output_dir", str(OUTPUT_DIR / case_name / "unet"),
            "--epochs", "10",
            "--batch_size", "4"
        ]
    elif model_type == "yolo":
        cmd = [
            sys.executable, "models/yolo/train.py",
            "--data_dir", str(train_dir),
            "--output_dir", str(OUTPUT_DIR / case_name / "yolo"),
            "--epochs", "10",
            "--batch_size", "4"
        ]
    elif model_type == "rf":
        cmd = [
            sys.executable, "models/random_forest/train.py",
            "--data_dir", str(train_dir),
            "--output_dir", str(OUTPUT_DIR / case_name / "rf"),
            "--epochs", "10"
        ]
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        if result.returncode == 0:
            print(f"  ✅ {model_type} training completed")
            return True
        else:
            print(f"  ❌ {model_type} training failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ⏰ {model_type} training timed out")
        return False
    except Exception as e:
        print(f"  ❌ {model_type} training error: {e}")
        return False

def evaluate_model(model_type, case_name):
    """Evaluate a trained model on the test set."""
    print(f"  Evaluating {model_type} model...")
    
    # Model-specific evaluation commands
    if model_type == "deeplab":
        cmd = [
            sys.executable, "models/deeplab/eval.py",
            "--model_path", str(OUTPUT_DIR / case_name / "deeplab" / "best_model.pth"),
            "--test_dir", str(TEST_SET),
            "--output_dir", str(OUTPUT_DIR / case_name / "deeplab" / "results")
        ]
    elif model_type == "unet":
        cmd = [
            sys.executable, "models/unet_no_patches/eval.py",
            "--model_path", str(OUTPUT_DIR / case_name / "unet" / "best_model.pth"),
            "--test_dir", str(TEST_SET),
            "--output_dir", str(OUTPUT_DIR / case_name / "unet" / "results")
        ]
    elif model_type == "yolo":
        cmd = [
            sys.executable, "models/yolo/eval.py",
            "--model_path", str(OUTPUT_DIR / case_name / "yolo" / "best_model.pt"),
            "--test_dir", str(TEST_SET),
            "--output_dir", str(OUTPUT_DIR / case_name / "yolo" / "results")
        ]
    elif model_type == "rf":
        cmd = [
            sys.executable, "models/random_forest/eval.py",
            "--model_path", str(OUTPUT_DIR / case_name / "rf" / "best_model.pkl"),
            "--test_dir", str(TEST_SET),
            "--output_dir", str(OUTPUT_DIR / case_name / "rf" / "results")
        ]
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout
        if result.returncode == 0:
            print(f"  ✅ {model_type} evaluation completed")
            return True
        else:
            print(f"  ❌ {model_type} evaluation failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ⏰ {model_type} evaluation timed out")
        return False
    except Exception as e:
        print(f"  ❌ {model_type} evaluation error: {e}")
        return False

def collect_metrics(case_name):
    """Collect metrics from evaluation results."""
    metrics = {}
    
    for model_type in ["deeplab", "unet", "yolo", "rf"]:
        results_dir = OUTPUT_DIR / case_name / model_type / "results"
        if results_dir.exists():
            # Look for metrics files
            metrics_file = results_dir / "metrics.json"
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r') as f:
                        model_metrics = json.load(f)
                        metrics[model_type] = model_metrics
                except:
                    metrics[model_type] = {"error": "Failed to parse metrics"}
            else:
                metrics[model_type] = {"error": "No metrics file found"}
        else:
            metrics[model_type] = {"error": "No results directory"}
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate pseudo-label experiments")
    parser.add_argument("--models", default="deeplab,unet,yolo,rf", 
                       help="Comma-separated list of models to evaluate")
    parser.add_argument("--case", help="Specific case to evaluate (e.g., '10_2_keep')")
    args = parser.parse_args()
    
    models_to_evaluate = args.models.split(",")
    
    print("🚀 Starting Pseudo-Label Evaluation")
    print("=" * 50)
    
    setup_directories()
    
    # Results storage
    all_results = {}
    
    # Generate all cases
    cases = []
    for img_count in IMAGE_COUNTS:
        for agreement in AGREEMENT_LEVELS:
            for handling in UNKNOWN_HANDLING:
                case_name = f"case_{img_count}_{agreement}_{handling}"
                cases.append((img_count, agreement, handling, case_name))
    
    # Filter to specific case if requested
    if args.case:
        cases = [case for case in cases if args.case in case[3]]
    
    total_cases = len(cases)
    print(f"📊 Evaluating {total_cases} cases:")
    for img_count, agreement, handling, case_name in cases:
        print(f"   - {img_count} images, K={agreement}, {handling} unknown pixels")
    
    print("\n" + "=" * 50)
    
    for i, (img_count, agreement, handling, case_name) in enumerate(cases, 1):
        print(f"\n🔍 Case {i}/{total_cases}: {case_name}")
        print(f"   Images: {img_count}, Agreement: K={agreement}, Unknown: {handling}")
        
        start_time = time.time()
        
        # Create training dataset
        print("  📁 Creating training dataset...")
        train_dir = create_training_dataset(img_count, agreement, handling)
        
        # Train models
        training_results = {}
        for model_type in models_to_evaluate:
            if train_model(model_type, train_dir, case_name):
                training_results[model_type] = "success"
            else:
                training_results[model_type] = "failed"
        
        # Evaluate models
        evaluation_results = {}
        for model_type in models_to_evaluate:
            if training_results[model_type] == "success":
                if evaluate_model(model_type, case_name):
                    evaluation_results[model_type] = "success"
                else:
                    evaluation_results[model_type] = "failed"
            else:
                evaluation_results[model_type] = "skipped"
        
        # Collect metrics
        metrics = collect_metrics(case_name)
        
        # Store results
        case_results = {
            "case_name": case_name,
            "img_count": img_count,
            "agreement": agreement,
            "unknown_handling": handling,
            "training_results": training_results,
            "evaluation_results": evaluation_results,
            "metrics": metrics,
            "execution_time": time.time() - start_time
        }
        
        all_results[case_name] = case_results
        
        # Save case results
        results_file = OUTPUT_DIR / case_name / "case_results.json"
        with open(results_file, 'w') as f:
            json.dump(case_results, f, indent=2)
        
        print(f"  ⏱️  Case completed in {case_results['execution_time']:.1f}s")
        
        # Clean up temporary files
        if train_dir.exists():
            shutil.rmtree(train_dir)
    
    # Save overall results
    overall_results_file = OUTPUT_DIR / "overall_results.json"
    with open(overall_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📋 EVALUATION SUMMARY")
    print("=" * 50)
    
    for case_name, results in all_results.items():
        print(f"\n{case_name}:")
        print(f"  Training: {results['training_results']}")
        print(f"  Evaluation: {results['evaluation_results']}")
        print(f"  Time: {results['execution_time']:.1f}s")
    
    print(f"\n✅ Evaluation complete! Results saved to {OUTPUT_DIR}")
    print(f"📊 Overall results: {overall_results_file}")

if __name__ == "__main__":
    main()
