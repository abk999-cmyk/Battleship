#!/usr/bin/env python3
"""
Battleship AI Research Platform - Complete Setup and Launch Script

This script handles the complete setup process and launches the dashboard:
- Validates Python 3.9 installation
- Creates/activates .venv39 virtual environment  
- Installs all dependencies from requirements.txt
- Sets up required directories
- Validates installation
- Launches the battleship dashboard

Usage:
    python3 setup_and_launch.py
    or
    ./setup_and_launch.py

Requirements:
    - Python 3.9 must be installed on the system
    - tkinter system dependencies (on some Linux systems)
"""

import os
import sys
import subprocess
import platform
import shutil
import logging
import traceback
import time
from pathlib import Path
from datetime import datetime


class BattleshipSetup:
    """Complete setup and launch manager for Battleship AI Research Platform"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.absolute()  # Go up one level from scripts/
        self.venv_name = ".venv39"
        self.venv_path = self.base_dir / self.venv_name
        self.logs_dir = self.base_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Required directories
        self.required_dirs = [
            'data', 'logs', 'models', 'reports', 'assets'
        ]
        
        # System info
        self.system = platform.system()
        self.is_windows = self.system == "Windows"
        self.python_executable = "python" if self.is_windows else "python3.9"
        
        self.logger.info(f"Initializing Battleship setup on {self.system}")
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_file = self.logs_dir / f"setup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Create logger
        self.logger = logging.getLogger('BattleshipSetup')
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Simple formatter for console
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging initialized - log file: {log_file}")
    
    def print_header(self):
        """Print setup header"""
        header = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                   Battleship AI Research Platform                            ║
║                     Complete Setup & Launch Script                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(header)
        self.logger.info("Starting Battleship AI Research Platform setup")
    
    def check_python_version(self):
        """Check and validate Python 3.9 installation"""
        self.logger.info("Checking Python version...")
        print("🔍 Checking Python version...")
        
        try:
            # Check current Python version
            major, minor = sys.version_info[:2]
            current_version = f"{major}.{minor}"
            
            self.logger.info(f"Current Python version: {current_version}")
            print(f"   Current Python: {current_version}")
            
            if major != 3 or minor != 9:
                self.logger.warning(f"Python 3.9 required, found Python {current_version}")
                print(f"⚠️  Warning: Python 3.9 is required for optimal compatibility.")
                print(f"   Found: Python {current_version}")
                print(f"   The project specifically requires Python 3.9 for:")
                print(f"   - tkinter compatibility")
                print(f"   - TensorFlow 2.17.0 compatibility")
                
                # Try to find Python 3.9
                python39_candidates = [
                    "python3.9",
                    "python39",
                    "/usr/bin/python3.9",
                    "/usr/local/bin/python3.9"
                ]
                
                python39_path = None
                for candidate in python39_candidates:
                    if shutil.which(candidate):
                        try:
                            result = subprocess.run(
                                [candidate, "--version"], 
                                capture_output=True, 
                                text=True, 
                                check=True
                            )
                            if "Python 3.9" in result.stdout:
                                python39_path = candidate
                                break
                        except Exception:
                            continue
                
                if python39_path:
                    print(f"✅ Found Python 3.9 at: {python39_path}")
                    self.python_executable = python39_path
                    self.logger.info(f"Using Python 3.9 at: {python39_path}")
                else:
                    print("❌ Python 3.9 not found. Please install Python 3.9:")
                    print("   macOS: brew install python@3.9")
                    print("   Ubuntu: sudo apt install python3.9 python3.9-venv python3.9-dev")
                    print("   Windows: Download from https://www.python.org/downloads/")
                    self.logger.error("Python 3.9 not found")
                    return False
            else:
                print("✅ Python 3.9 detected - perfect!")
                self.logger.info("Python 3.9 confirmed")
            
            # Test tkinter availability
            try:
                import tkinter
                print("✅ tkinter is available")
                self.logger.info("tkinter module available")
            except ImportError:
                print("❌ tkinter not available. Install with:")
                if self.system == "Darwin":  # macOS
                    print("   If using Homebrew: brew install python-tk")
                elif self.system == "Linux":
                    print("   Ubuntu/Debian: sudo apt-get install python3-tk")
                    print("   CentOS/RHEL: sudo yum install tkinter")
                self.logger.error("tkinter not available")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking Python version: {e}")
            print(f"❌ Error checking Python version: {e}")
            return False
    
    def setup_virtual_environment(self):
        """Create and setup Python 3.9 virtual environment"""
        self.logger.info("Setting up virtual environment...")
        print("🔧 Setting up virtual environment...")
        
        try:
            # Remove existing venv if it exists
            if self.venv_path.exists():
                print(f"   Removing existing virtual environment...")
                self.logger.info("Removing existing virtual environment")
                shutil.rmtree(self.venv_path)
            
            # Create new virtual environment
            print(f"   Creating new virtual environment: {self.venv_name}")
            self.logger.info(f"Creating virtual environment: {self.venv_name}")
            
            cmd = [self.python_executable, "-m", "venv", str(self.venv_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            print("✅ Virtual environment created successfully")
            self.logger.info("Virtual environment created successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to create virtual environment: {e}")
            self.logger.error(f"Command output: {e.stdout}")
            self.logger.error(f"Command stderr: {e.stderr}")
            print(f"❌ Failed to create virtual environment: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error setting up virtual environment: {e}")
            print(f"❌ Error setting up virtual environment: {e}")
            return False
    
    def get_venv_python(self):
        """Get the Python executable path in the virtual environment"""
        if self.is_windows:
            return self.venv_path / "Scripts" / "python.exe"
        else:
            return self.venv_path / "bin" / "python"
    
    def get_venv_pip(self):
        """Get the pip executable path in the virtual environment"""
        if self.is_windows:
            return self.venv_path / "Scripts" / "pip.exe"
        else:
            return self.venv_path / "bin" / "pip"
    
    def install_dependencies(self):
        """Install dependencies from requirements.txt"""
        self.logger.info("Installing dependencies...")
        print("📦 Installing dependencies...")
        
        try:
            requirements_file = self.base_dir / "requirements.txt"
            if not requirements_file.exists():
                self.logger.error("requirements.txt not found")
                print("❌ requirements.txt not found")
                return False
            
            venv_pip = self.get_venv_pip()
            
            # Upgrade pip first
            print("   Upgrading pip...")
            cmd = [str(venv_pip), "install", "--upgrade", "pip"]
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Install requirements
            print("   Installing packages from requirements.txt...")
            self.logger.info("Installing packages from requirements.txt")
            
            cmd = [str(venv_pip), "install", "-r", str(requirements_file)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            print("✅ Dependencies installed successfully")
            self.logger.info("Dependencies installed successfully")
            
            # Log installed packages
            cmd = [str(venv_pip), "list"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.logger.debug(f"Installed packages:\n{result.stdout}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install dependencies: {e}")
            self.logger.error(f"Command output: {e.stdout}")
            self.logger.error(f"Command stderr: {e.stderr}")
            print(f"❌ Failed to install dependencies")
            print(f"   Error: {e}")
            if e.stderr:
                print(f"   Details: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"Error installing dependencies: {e}")
            print(f"❌ Error installing dependencies: {e}")
            return False
    
    def setup_directories(self):
        """Create required directories"""
        self.logger.info("Setting up directories...")
        print("📁 Setting up directories...")
        
        try:
            for directory in self.required_dirs:
                dir_path = self.base_dir / directory
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)
                    print(f"   Created: {directory}/")
                    self.logger.info(f"Created directory: {directory}")
                else:
                    print(f"   Exists: {directory}/")
                    self.logger.debug(f"Directory already exists: {directory}")
            
            print("✅ Directories setup complete")
            self.logger.info("Directories setup complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up directories: {e}")
            print(f"❌ Error setting up directories: {e}")
            return False
    
    def validate_installation(self):
        """Validate the installation by testing imports"""
        self.logger.info("Validating installation...")
        print("🧪 Validating installation...")
        
        try:
            venv_python = self.get_venv_python()
            
            # Test critical imports
            test_imports = [
                "import numpy",
                "import pandas", 
                "import matplotlib",
                "import tensorflow",
                "import tkinter",
                "import PIL",
                "import sklearn"
            ]
            
            for test_import in test_imports:
                package_name = test_import.split()[1]
                print(f"   Testing {package_name}...", end="")
                
                cmd = [str(venv_python), "-c", test_import]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(" ✅")
                    self.logger.debug(f"Import test passed: {package_name}")
                else:
                    print(" ❌")
                    self.logger.warning(f"Import test failed: {package_name}")
                    self.logger.warning(f"Error: {result.stderr}")
            
            # Test if main files exist
            critical_files = [
                "apps/battleship_dashboard.py",
                "core/game.py", 
                "core/board.py",
                "core/player.py",
                "agents/AI_agent.py"
            ]
            
            print("   Checking critical files...")
            for file_name in critical_files:
                file_path = self.base_dir / file_name
                if file_path.exists():
                    print(f"     {file_name} ✅")
                    self.logger.debug(f"Critical file found: {file_name}")
                else:
                    print(f"     {file_name} ❌")
                    self.logger.warning(f"Critical file missing: {file_name}")
            
            print("✅ Installation validation complete")
            self.logger.info("Installation validation complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating installation: {e}")
            print(f"❌ Error validating installation: {e}")
            return False
    
    def launch_dashboard(self):
        """Launch the battleship dashboard"""
        self.logger.info("Launching dashboard...")
        print("🚀 Launching Battleship AI Research Dashboard...")
        
        try:
            dashboard_script = self.base_dir / "apps" / "battleship_dashboard.py"
            if not dashboard_script.exists():
                self.logger.error("apps/battleship_dashboard.py not found")
                print("❌ apps/battleship_dashboard.py not found")
                return False
            
            venv_python = self.get_venv_python()
            
            print("   Starting dashboard application...")
            print("   (This will open the GUI window)")
            
            # Change to project directory and run
            os.chdir(self.base_dir)
            
            cmd = [str(venv_python), str(dashboard_script)]
            self.logger.info(f"Executing: {' '.join(cmd)}")
            
            # Run the dashboard
            subprocess.run(cmd, check=True)
            
            self.logger.info("Dashboard launched successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Dashboard failed with return code {e.returncode}")
            print(f"❌ Dashboard failed to start")
            return False
        except KeyboardInterrupt:
            self.logger.info("Dashboard interrupted by user")
            print("\n🛑 Dashboard stopped by user")
            return True
        except Exception as e:
            self.logger.error(f"Error launching dashboard: {e}")
            print(f"❌ Error launching dashboard: {e}")
            return False
    
    def run_complete_setup(self):
        """Run the complete setup and launch process"""
        try:
            self.print_header()
            
            # Step 1: Check Python version
            if not self.check_python_version():
                print("\n❌ Setup failed: Python 3.9 requirement not met")
                return False
            
            # Step 2: Setup virtual environment  
            if not self.setup_virtual_environment():
                print("\n❌ Setup failed: Virtual environment creation failed")
                return False
            
            # Step 3: Install dependencies
            if not self.install_dependencies():
                print("\n❌ Setup failed: Dependency installation failed")
                return False
            
            # Step 4: Setup directories
            if not self.setup_directories():
                print("\n❌ Setup failed: Directory setup failed")  
                return False
            
            # Step 5: Validate installation
            if not self.validate_installation():
                print("\n⚠️  Setup completed with warnings: Some validation tests failed")
                print("   The application may still work, continuing to launch...")
            
            print("\n✅ Setup completed successfully!")
            print("="*80)
            
            # Step 6: Launch dashboard
            return self.launch_dashboard()
            
        except KeyboardInterrupt:
            print("\n\n🛑 Setup interrupted by user")
            self.logger.info("Setup interrupted by user")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during setup: {e}")
            self.logger.error(traceback.format_exc())
            print(f"\n❌ Unexpected error during setup: {e}")
            return False


def main():
    """Main entry point"""
    setup = BattleshipSetup()
    
    try:
        success = setup.run_complete_setup()
        
        if success:
            print("\n🎉 Battleship AI Research Platform is ready!")
            print("   Re-run this script anytime to launch the dashboard")
        else:
            print("\n💡 Troubleshooting tips:")
            print("   1. Ensure Python 3.9 is installed")
            print("   2. Check that tkinter is available")  
            print("   3. Verify internet connection for package downloads")
            print("   4. Check the log files in logs/ directory")
            print(f"   5. Latest log: {setup.logs_dir}")
            
        return 0 if success else 1
        
    except Exception as e:
        setup.logger.error(f"Fatal error in main: {e}")
        setup.logger.error(traceback.format_exc())
        print(f"\n💥 Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
