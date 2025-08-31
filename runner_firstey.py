import os
import sys
import cv2
import ffmpeg

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_frames.py <input_video>")
        sys.exit(1)

    video_path = sys.argv[1]

    if not os.path.isfile(video_path):
        print(f"❌ File not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open the video file.")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0
    cap.release()

    print(f"🎥 Input Video: {video_path}")
    print(f"📊 FPS: {fps}")
    print(f"🖼 Total Frames: {total_frames}")
    print(f"⏱ Duration: {duration_sec:.2f} sec ({duration_sec/60:.2f} min)")

    frames_dir = "E:\mini project\ECCV2022-RIFE\frames"
    os.makedirs(frames_dir, exist_ok=True)

    print(f"📤 Extracting frames to: {frames_dir} ...")
    (
        ffmpeg
        .input(video_path)
        .output(f'{frames_dir}/frame_%06d.png', start_number=0)
        .run(capture_stdout=True, capture_stderr=True)
    )

    print(f"✅ Done! Extracted frames saved to: {frames_dir}")

if __name__ == "__main__":
    main()
