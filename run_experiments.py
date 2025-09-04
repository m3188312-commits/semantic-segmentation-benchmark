#!/usr/bin/env python3
"""
Simple runner script for pseudo-labeling experiments.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the experiments with different configurations."""
    
    print("🚀 Starting Pseudo-Labeling Experiments")
    print("="*60)
    
    # Basic configuration
    epochs = 50  # Adjust as needed
    batch_size = 4
    lr = 1e-4
    
    # Quick test run (fewer epochs for testing)
    if "--test" in sys.argv:
        epochs = 5
        print("⚡ Running in TEST MODE (5 epochs only)")
    
    # Full run
    cmd = [
        "python", "pseudo_labeling_experiments.py",
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--output", "experiment_results.csv"
    ]
    
    if "--skip-baseline" in sys.argv:
        cmd.append("--skip_baseline")
        
    print(f"Running command: {' '.join(cmd)}")
    print("="*60)
    
    # Run the experiments
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ All experiments completed successfully!")
        
        # Show results file
        results_file = Path("experiment_results.csv")
        if results_file.exists():
            print(f"📊 Results saved to: {results_file}")
            print(f"📂 Model checkpoints saved to: checkpoints/")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Experiments failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n⚠️  Experiments interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage:")
        print("  python run_experiments.py              # Full run (50 epochs)")
        print("  python run_experiments.py --test       # Quick test (5 epochs)")
        print("  python run_experiments.py --skip-baseline  # Skip baseline")
    else:
        main()
