#!/usr/bin/env python
import os
import sys
import platform
import subprocess
from pathlib import Path
import zipfile
import shutil
import urllib.request

def download_file(url, target_path):
    """Download a file from URL to target path with progress indicator"""
    print(f"Downloading {url} to {target_path}...")
    
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(int(downloaded * 100 / total_size), 100)
        sys.stdout.write(f"\rProgress: {percent}% ({downloaded/1024/1024:.1f}MB/{total_size/1024/1024:.1f}MB)")
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, target_path, reporthook=report_progress)
    print("\nDownload complete!")

def setup_ffmpeg():
    """Download and setup FFmpeg"""
    current_dir = Path(__file__).parent.absolute()
    
    if platform.system() == "Windows":
        ffmpeg_exe_path = current_dir / "ffmpeg.exe"
        if ffmpeg_exe_path.exists():
            print("FFmpeg already exists.")
            return
        
        ffmpeg_zip = current_dir / "ffmpeg.zip"
        download_file(
            "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip", 
            ffmpeg_zip
        )
        
        # Extract only ffmpeg.exe from the zip
        print("Extracting FFmpeg...")
        with zipfile.ZipFile(ffmpeg_zip, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if file.endswith("ffmpeg.exe"):
                    zip_ref.extract(file, current_dir)
                    extracted_path = current_dir / file
                    shutil.move(extracted_path, ffmpeg_exe_path)
                    break
        
        # Clean up
        os.remove(ffmpeg_zip)
        ffmpeg_dir = current_dir / "ffmpeg-master-latest-win64-gpl"
        if ffmpeg_dir.exists():
            shutil.rmtree(ffmpeg_dir)
        
        print(f"FFmpeg executable is now available at: {ffmpeg_exe_path}")
    
    elif platform.system() == "Linux":
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("FFmpeg is already installed.")
        except FileNotFoundError:
            print("Installing FFmpeg...")
            subprocess.run(["sudo", "apt", "update"], check=True)
            subprocess.run(["sudo", "apt", "install", "-y", "ffmpeg"], check=True)
            print("FFmpeg installed successfully.")
    
    elif platform.system() == "Darwin":  # macOS
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("FFmpeg is already installed.")
        except FileNotFoundError:
            print("Installing FFmpeg using Homebrew...")
            try:
                # Check if Homebrew is installed
                subprocess.run(["brew", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except FileNotFoundError:
                print("Homebrew is not installed. Please install Homebrew first:")
                print("/bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
                sys.exit(1)
            
            subprocess.run(["brew", "install", "ffmpeg"], check=True)
            print("FFmpeg installed successfully.")

def setup_cuda():
    """Check CUDA availability and provide instructions if not available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
            print(f"PyTorch version: {torch.__version__}")
            return True
        else:
            print("CUDA is not available. Using CPU mode.")
            print("\nTo enable GPU acceleration, please follow these steps:")
            print("1. Install the latest NVIDIA drivers for your GPU")
            print("2. Install CUDA Toolkit 11.8 from: https://developer.nvidia.com/cuda-11-8-0-download-archive")
            print("3. Install PyTorch with CUDA support: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            return False
    except ImportError:
        print("PyTorch is not installed. Installing required packages...")
        return False

def download_model_weights():
    """Download pre-trained model weights if they don't exist"""
    current_dir = Path(__file__).parent.absolute()
    model_dir = current_dir / "train_log"
    model_file = model_dir / "flownet.pkl"
    
    if model_file.exists():
        print("Model weights already exist.")
        return
    
    model_dir.mkdir(exist_ok=True)
    
    print("Downloading model weights...")
    download_file(
        "https://github.com/hzwer/Practical-RIFE/releases/download/1.0/flownet.pkl",
        model_file
    )
    print(f"Model weights downloaded to {model_file}")

def main():
    print("Setting up interpoGen environment...")
    
    # Install Python dependencies
    print("\n1. Installing Python dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Setup external dependencies
    print("\n2. Setting up FFmpeg...")
    setup_ffmpeg()
    
    # Check CUDA
    print("\n3. Checking CUDA availability...")
    setup_cuda()
    
    # Download model weights
    print("\n4. Setting up model weights...")
    download_model_weights()
    
    print("\nSetup complete! You can now run the interpoGen tools.")

if __name__ == "__main__":
    main()
