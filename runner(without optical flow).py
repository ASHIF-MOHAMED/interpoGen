import os
import sys
import cv2
import numpy as np
import torch
import csv
import time
from skimage.metrics import structural_similarity as ssim

def pixel_diff(frame1, frame2):
    """Mean absolute difference between two frames"""
    t1 = torch.from_numpy(frame1).to(torch.float32).to('cuda')
    t2 = torch.from_numpy(frame2).to(torch.float32).to('cuda')
    diff = torch.abs(t1 - t2)
    return diff.mean().item()

def histogram_diff(frame1, frame2):
    """Compare color histograms (Bhattacharyya distance)"""
    hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8],
                         [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8],
                         [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

def ssim_diff(frame1, frame2):
    """Stable SSIM using skimage"""
    # Resize to same size if mismatch
    if frame1.shape != frame2.shape:
        min_h = min(frame1.shape[0], frame2.shape[0])
        min_w = min(frame1.shape[1], frame2.shape[1])
        frame1 = frame1[:min_h, :min_w]
        frame2 = frame2[:min_h, :min_w]

    # Convert to grayscale for SSIM
    f1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    f2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    try:
        score, _ = ssim(f1, f2, full=True, data_range=255)
        if np.isnan(score):
            return 0.0
        return float(score)
    except Exception:
        return 0.0

def compute_metrics(frame1, frame2):
    """Compute PixelDiff, SSIM, HistogramDiff"""
    pdiff = pixel_diff(frame1, frame2)
    ssim_val = ssim_diff(frame1, frame2)
    hdiff = histogram_diff(frame1, frame2)
    return pdiff, ssim_val, hdiff

def main():
    if len(sys.argv) < 3:
        print("Usage: python runner1.py <video_path> <output_directory>")
        sys.exit(1)

    video_path = sys.argv[1].strip('"')
    frames_dir = sys.argv[2].strip('"')

    if not os.path.isfile(video_path):
        print(f"File not found: {video_path}")
        sys.exit(1)

    print(f"frames_dir: {repr(frames_dir)}")
    os.makedirs(frames_dir, exist_ok=True)
    start_time = time.time()
    print(f"Starting extraction at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    save_frames = False  # Set to True if you want to save frames as images

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open the video file.")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")

    # CSV to log metrics
    csv_path = os.path.join(frames_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Frame_i", "Frame_(i+1)", "PixelDiff", "SSIM", "HistDiff"])

        ret, prev_frame = cap.read()
        if not ret:
            print("Failed to read first frame")
            sys.exit(1)

        frame_idx = 0
        print(f"Extracting frame {frame_idx}")
        if save_frames:
            cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_idx:06d}.png"), prev_frame)

        while True:
            step_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            print(f"Extracting frame {frame_idx}")
            if save_frames:
                cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_idx:06d}.png"), frame)

            # Compute metrics between consecutive frames
            metrics_start = time.time()
            pdiff, ssim_val, hdiff = compute_metrics(prev_frame, frame)
            metrics_end = time.time()
            writer.writerow([frame_idx - 1, frame_idx, pdiff, ssim_val, hdiff])

            print(f"Frame {frame_idx-1}-{frame_idx} | PixelDiff={pdiff:.4f}, SSIM={ssim_val:.4f}, HistDiff={hdiff:.4f} | Metrics time: {metrics_end-metrics_start:.3f}s | Step time: {metrics_end-step_start:.3f}s")

            prev_frame = frame

    cap.release()
    end_time = time.time()
    print(f"Ending extraction at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    duration = end_time - start_time
    print(f"Total extraction duration: {duration:.2f} seconds ({duration/60:.2f} min)")
    print(f"Frames + metrics extracted to '{frames_dir}'")

if __name__ == "__main__":
    main()
