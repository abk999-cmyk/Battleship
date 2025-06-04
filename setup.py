"""
Installation script for the Battleship AI Research Platform
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

# Directory setup
BASE_DIR = Path(__file__).parent
REQUIRED_DIRS = [
    'data',
    'logs',
    'models',
    'reports',
    'assets'
]

# Required packages
REQUIRED_PACKAGES = [
    'numpy',
    'pandas',
    'matplotlib',
    'seaborn',
    'tensorflow',
    'scikit-learn',
    'networkx',
    'scipy',
    'customtkinter',
    'tqdm',
    'pillow'
]


def print_header():
    """Print installation header"""
    print("\n" + "=" * 80)
    print("Battleship AI Research Platform - Installation")
    print("=" * 80 + "\n")


def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 8):
        print(f"Error: Python 3.8 or higher is required. You have Python {major}.{minor}.")
        return False

    print(f"✓ Python {major}.{minor} detected. Compatible version.")
    return True


def create_directories():
    """Create required directories if they don't exist"""
    print("Setting up directories...")
    for directory in REQUIRED_DIRS:
        dir_path = BASE_DIR / directory
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
            print(f"  Created directory: {directory}")
        else:
            print(f"  Directory already exists: {directory}")


def install_dependencies():
    """Install required Python packages"""
    print("\nInstalling required packages...")

    # Check if pip is available
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "--version"],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("Error: pip is not available. Please install pip first.")
        return False

    # Install each package
    for package in REQUIRED_PACKAGES:
        print(f"  Installing {package}...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"    ✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"    × Failed to install {package}. Error: {e}")
            print(f"      Try manual installation: pip install {package}")

    print("\nAll dependencies installed.")
    return True


def create_sample_assets():
    """Create sample assets if they don't exist"""
    print("\nSetting up game assets...")
    assets_dir = BASE_DIR / 'assets'

    # Sample ship silhouette
    try:
        from PIL import Image, ImageDraw

        # Create water tile
        water_image = Image.new('RGBA', (48, 48), (30, 144, 255, 255))
        # Add wave pattern
        draw = ImageDraw.Draw(water_image)
        for y in range(0, 48, 6):
            for x in range(0, 48, 12):
                draw.arc([x - 2, y - 2, x + 6, y + 6], 0, 180, fill=(60, 170, 255, 128), width=1)
        water_image.save(assets_dir / 'water.png')
        print("  Created water.png")

        # Create ship tile
        ship_image = Image.new('RGBA', (48, 48), (0, 0, 0, 0))
        draw = ImageDraw.Draw(ship_image)
        # Ship body
        draw.rectangle([8, 8, 40, 40], fill=(128, 128, 128, 255), outline=(80, 80, 80, 255), width=2)
        # Deck features
        draw.rectangle([16, 16, 32, 32], fill=(100, 100, 100, 255), outline=(60, 60, 60, 255), width=1)
        ship_image.save(assets_dir / 'ship.png')
        print("  Created ship.png")

        # Create hit marker
        hit_image = Image.new('RGBA', (48, 48), (0, 0, 0, 0))
        draw = ImageDraw.Draw(hit_image)
        # Explosion
        draw.ellipse([10, 10, 38, 38], fill=(255, 76, 76, 200), outline=(200, 0, 0, 255), width=2)
        # Crosshairs
        draw.line([24, 10, 24, 38], fill=(200, 0, 0, 255), width=2)
        draw.line([10, 24, 38, 24], fill=(200, 0, 0, 255), width=2)
        hit_image.save(assets_dir / 'hit.png')
        print("  Created hit.png")

        # Create miss marker
        miss_image = Image.new('RGBA', (48, 48), (0, 0, 0, 0))
        draw = ImageDraw.Draw(miss_image)
        # Water splash
        draw.ellipse([14, 14, 34, 34], fill=(76, 114, 176, 200), outline=(50, 80, 150, 255), width=2)
        # Ripples
        draw.arc([8, 8, 40, 40], 0, 360, fill=(50, 80, 150, 200), width=1)
        miss_image.save(assets_dir / 'miss.png')
        print("  Created miss.png")

        print("  ✓ Game assets created successfully")

    except Exception as e:
        print(f"  × Error creating game assets: {e}")
        print("    You can still run the game without assets")


def test_installation():
    """Run basic tests to verify installation"""
    print("\nTesting installation...")

    try:
        # Test numpy
        print("  Testing numpy...", end="")
        import numpy as np
        np.random.rand(3, 3)
        print(" ✓")

        # Test pandas
        print("  Testing pandas...", end="")
        import pandas as pd
        pd.DataFrame({'test': [1, 2, 3]})
        print(" ✓")

        # Test matplotlib
        print("  Testing matplotlib...", end="")
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        plt.figure()
        plt.close()
        print(" ✓")

        # Test tensorflow
        print("  Testing tensorflow...", end="")
        import tensorflow as tf
        tf.constant([1, 2, 3])
        print(" ✓")

        # Test customtkinter
        print("  Testing customtkinter...", end="")
        import customtkinter as ctk
        ctk.deactivate_automatic_dpi_awareness()
        print(" ✓")

        print("\n✓ All tests passed! Installation successful.")
        return True

    except ImportError as e:
        print(f"\n× Error: {e}")
        print("  Installation may be incomplete. Try installing the missing package manually.")
        return False
    except Exception as e:
        print(f"\n× Error: {e}")
        print("  Installation may have issues. Please check the error message.")
        return False


def create_launch_script():
    """Create a launch script for easy startup"""
    print("\nCreating launch script...")

    # Determine the correct extension based on OS
    if platform.system() == "Windows":
        script_name = "launch_battleship.bat"
        script_content = "@echo off\n" \
                         f"cd {BASE_DIR}\n" \
                         f"python battleship_dashboard.py\n" \
                         "pause\n"
    else:
        script_name = "launch_battleship.sh"
        script_content = "#!/bin/bash\n" \
                         f"cd {BASE_DIR}\n" \
                         f"python3 battleship_dashboard.py\n"

    script_path = BASE_DIR / script_name
    with open(script_path, 'w') as f:
        f.write(script_content)

    # Make the script executable on Unix-like systems
    if platform.system() != "Windows":
        os.chmod(script_path, 0o755)

    print(f"✓ Created launch script: {script_name}")
    print(f"  Run this script to start the Battleship AI Research Platform")


def main():
    """Main installation function"""
    print_header()

    if not check_python_version():
        sys.exit(1)

    create_directories()

    if not install_dependencies():
        print("\nWarning: Some dependencies may not have been installed correctly.")
        print("You may need to install them manually.")

    create_sample_assets()

    if test_installation():
        create_launch_script()

        print("\n" + "=" * 80)
        print("Installation complete! You can now run the Battleship AI Research Platform.")
        print("=" * 80 + "\n")
    else:
        print("\n" + "=" * 80)
        print("Installation completed with some issues. Please fix them before running the platform.")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()