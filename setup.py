import os
import subprocess
import sys
import venv
from pathlib import Path

def main():
    # Define the venv directory
    venv_dir = Path("venv")
    
    # Create venv if it doesn't exist
    if not venv_dir.exists():
        print("Creating virtual environment...")
        venv.create(venv_dir, with_pip=True)
    
    # Determine the path to the activated Python executable
    if sys.platform == "win32":
        python_path = venv_dir / "Scripts" / "python.exe"
        pip_path = venv_dir / "Scripts" / "pip.exe"
    else:  # Unix-like systems (Linux, macOS)
        python_path = venv_dir / "bin" / "python"
        pip_path = venv_dir / "bin" / "pip"

    # Upgrade pip
    print("Upgrading pip...")
    subprocess.check_call([str(python_path), "-m", "pip", "install", "--upgrade", "pip"])

    # Install requirements
    print("Installing requirements...")
    requirements_path = Path("requirements.txt")
    if requirements_path.exists():
        subprocess.check_call([str(pip_path), "install", "-r", str(requirements_path)])
    else:
        print("requirements.txt not found!")
        return

    print("\nSetup completed successfully!")
    print("\nTo activate the virtual environment:")
    if sys.platform == "win32":
        print("    .\\venv\\Scripts\\activate")
    else:
        print("    source venv/bin/activate")

if __name__ == "__main__":
    main()
