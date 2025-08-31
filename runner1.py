import os
import sys
import cv2
import subprocess
import platform

def find_ffmpeg():
    """Find the FFmpeg executable path"""
    # First check if the local ffmpeg exists
    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_ffmpeg = os.path.join(current_dir, "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg")
    
    if os.path.isfile(local_ffmpeg):
        return local_ffmpeg
    
    # Try system ffmpeg
    try:
        if platform.system() == "Windows":
            # Check if ffmpeg is in PATH
            result = subprocess.run(["where", "ffmpeg"], capture_output=True, text=True)
            if result.returncode == 0:
                return "ffmpeg"
        else:
            result = subprocess.run(["which", "ffmpeg"], capture_output=True, text=True)
            if result.returncode == 0:
                return "ffmpeg"
    except Exception:
        pass
    
    # If we couldn't find ffmpeg, suggest installation
    print("FFmpeg not found. Please run setup.py to install dependencies or install FFmpeg manually.")
    print("For manual installation, visit: https://ffmpeg.org/download.html")
    sys.exit(1)

def main():
    if len(sys.argv) < 3:
        print("Usage: python runner1.py <video_path> <output_directory>")
        sys.exit(1)

    # Fix potential issues with spaces in the path
    video_path = sys.argv[1].strip('"')
    frames_dir = sys.argv[2].strip('"')

    if not os.path.isfile(video_path):
        print(f"File not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open the video file.")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0
    cap.release()
    print(f"FPS: {fps}")
    print(f"total frames {total_frames}")
    print(f"Duration: {duration_sec:.2f} sec ({duration_sec/60:.2f} min)")

    os.makedirs(frames_dir, exist_ok=True)

    try:
        ffmpeg_path = find_ffmpeg()
        output_pattern = os.path.join(frames_dir, 'frame_%06d.png')
        cmd = [ffmpeg_path, '-i', video_path, '-start_number', '0', output_pattern, '-y']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("FFmpeg error:", result.stderr)
            sys.exit(1)
    except Exception as e:
        print("Error:", str(e))
        sys.exit(1)
    print(f"Frames successfully extracted to '{frames_dir}'")
if __name__ == "__main__":
    main()
