"""
TensorFlow Setup Script for Battleship AI

This script installs the correct version of TensorFlow and dependencies for the
Battleship AI project, ensuring compatibility between packages.
"""

import sys
import subprocess
import platform
import os


def check_python_version():
    """Check if Python version is compatible with TensorFlow"""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 7):
        print(f"Error: Python 3.7 or higher is required for TensorFlow. You have Python {major}.{minor}.")
        print("Please install a compatible Python version.")
        return False

    print(f"✓ Python {major}.{minor} detected. Compatible with TensorFlow.")
    return True


def install_tensorflow():
    """Install the appropriate TensorFlow version based on platform"""
    system = platform.system()
    is_apple_silicon = False

    # Check if running on Apple Silicon
    if system == "Darwin" and platform.machine() == "arm64":
        is_apple_silicon = True
        print("Detected Apple Silicon Mac")

    print("\nInstalling TensorFlow and dependencies...")

    # Common dependencies
    common_deps = [
        "numpy==1.19.5",
        "scipy==1.7.3",
        "matplotlib==3.5.1",
        "tqdm==4.64.0"
    ]

    # Install common dependencies first
    for dep in common_deps:
        print(f"  Installing {dep}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
        except subprocess.CalledProcessError:
            print(f"  Warning: Failed to install {dep}, continuing anyway...")

    # Install the appropriate TensorFlow version
    if is_apple_silicon:
        # For Apple Silicon, we need special installation steps
        print("\nInstalling TensorFlow for Apple Silicon Mac...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install",
                                   "tensorflow-macos==2.17.0", "tensorflow-metal==1.3.0"])
            print("✓ TensorFlow for Apple Silicon installed successfully")
        except subprocess.CalledProcessError:
            print("× Failed to install TensorFlow for Apple Silicon")
            return False
    else:
        # For other platforms
        print("\nInstalling TensorFlow...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow==2.17.0"])
            print("✓ TensorFlow installed successfully")
        except subprocess.CalledProcessError:
            print("× Failed to install TensorFlow")

    # Additional dependencies for full functionality
    additional_deps = [
        "pandas==1.3.5",
        "scikit-learn==1.0.2",
        "networkx==2.7.1",
        "pillow==9.0.1",
        "customtkinter==5.1.3"  # Or latest version if not available
    ]

    print("\nInstalling additional dependencies...")
    for dep in additional_deps:
        print(f"  Installing {dep}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
        except subprocess.CalledProcessError:
            print(f"  Warning: Failed to install {dep}, continuing anyway...")

    return True


def verify_tensorflow():
    """Verify TensorFlow installation"""
    print("\nVerifying TensorFlow installation...")

    try:
        print("  Importing TensorFlow...")
        import tensorflow as tf
        print(f"  ✓ TensorFlow {tf.__version__} imported successfully")

        # Print GPU availability
        if tf.config.list_physical_devices('GPU'):
            print("  ✓ GPU is available for TensorFlow acceleration")
            print("    Detected devices:")
            for device in tf.config.list_physical_devices():
                print(f"    - {device}")
        else:
            print("  ⚠ No GPU detected. TensorFlow will run on CPU only.")

        # Test basic TensorFlow operation
        print("  Testing TensorFlow operation...")
        tensor = tf.constant([[1, 2], [3, 4]])
        result = tf.matmul(tensor, tensor)
        print("  ✓ Basic TensorFlow operation succeeded")

        return True
    except ImportError as e:
        print(f"  × Failed to import TensorFlow: {e}")
        return False
    except Exception as e:
        print(f"  × Error testing TensorFlow: {e}")
        return False


def fix_battleship_ai_path():
    """Create a directory for model files if it doesn't exist"""
    print("\nPreparing model directories...")

    dirs = ['models', 'data', 'logs', 'reports', 'assets']
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"  Created directory: {directory}")
        else:
            print(f"  Directory already exists: {directory}")

    return True


def main():
    """Main setup function"""
    print("=" * 70)
    print("Battleship AI TensorFlow Setup")
    print("=" * 70)

    if not check_python_version():
        return 1

    if not install_tensorflow():
        print("\nTensorFlow installation failed. Please check error messages above.")
        return 1

    if not verify_tensorflow():
        print("\nTensorFlow verification failed. The installation might be incomplete.")
        print("Please try manually installing TensorFlow with:")
        print("  pip install tensorflow==2.17.0")
        return 1

    if not fix_battleship_ai_path():
        print("\nFailed to prepare model directories.")
        return 1

    print("\n" + "=" * 70)
    print("Setup completed successfully!")
    print("You can now run the Battleship AI dashboard with the neural models.")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())