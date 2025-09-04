import os
import sys
import cv2
import numpy as np
import torch
import torchvision
import csv

def pixel_diff(frame1, frame2):
    """Mean absolute difference between two frames"""
    t1 = torch.from_numpy(frame1).float().to('cpu')
    t2 = torch.from_numpy(frame2).float().to('cpu')
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

def optical_flow_farnebäck(frame1, frame2):
    """Compute optical flow magnitude using Farnebäck (GPU if available, else CPU)"""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    try:
        gpu_gray1 = cv2.cuda_GpuMat()
        gpu_gray2 = cv2.cuda_GpuMat()
        gpu_gray1.upload(gray1)
        gpu_gray2.upload(gray2)

        farneback = cv2.cuda_FarnebackOpticalFlow.create(
            numLevels=5,
            pyrScale=0.5,
            fastPyramids=False,
            winSize=15,
            numIters=3,
            polyN=5,
            polySigma=1.2,
            flags=0
        )

        flow = farneback.calc(gpu_gray1, gpu_gray2, None)
        flow_cpu = flow.download()

    except Exception:
        # Fall back to CPU
        flow_cpu = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            0.5, 5, 15, 3, 5, 1.2, 0
        )

    magnitude, _ = cv2.cartToPolar(flow_cpu[..., 0], flow_cpu[..., 1])
    return float(np.mean(magnitude))

def compute_metrics(frame1, frame2):
    """Compute PixelDiff, SSIM, HistogramDiff, OpticalFlow"""
    t1 = torch.from_numpy(frame1).float().to('cpu') / 255.0
    t2 = torch.from_numpy(frame2).float().to('cpu') / 255.0
    # Ensure shape is (N, C, H, W)
    if t1.ndim == 3:
        t1 = t1.permute(2, 0, 1).unsqueeze(0)
    if t2.ndim == 3:
        t2 = t2.permute(2, 0, 1).unsqueeze(0)

    pdiff = pixel_diff(frame1, frame2)
    ssim_val = float('nan')
    try:
        ssim_val = torchvision.metrics.structural_similarity_index_measure(t1, t2).item()
    except Exception:
        # Fallback to skimage if torchvision fails
        try:
            from skimage.metrics import structural_similarity as skimage_ssim
            # Convert to grayscale for skimage
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            ssim_val = skimage_ssim(gray1, gray2, data_range=gray2.max() - gray2.min())
        except Exception:
            ssim_val = float('nan')
    hdiff = histogram_diff(frame1, frame2)
    oflow = optical_flow_farnebäck(frame1, frame2)

    return pdiff, ssim_val, hdiff, oflow

# Classify frame pair type based on metrics
def classify_frame(pdiff, ssim_val, hdiff, oflow):
    # Prioritize scene change detection
    if pdiff >= 40 or ssim_val < 0.90 or hdiff >= 0.2 or oflow >= 1.0:
        return "scene_change"
    # Identical frames (very strict)
    if pdiff < 2 and ssim_val > 0.99 and hdiff < 0.01 and oflow < 0.05:
        return "identical"
    # Very little motion (any metric in range)
    if (2 <= pdiff < 10) or (0.97 <= ssim_val <= 0.99) or (0.01 <= hdiff < 0.05) or (0.05 <= oflow < 0.2):
        return "very_little_motion"
    # Normal motion (any metric in range)
    if (10 <= pdiff < 40) or (0.90 <= ssim_val < 0.97) or (0.05 <= hdiff < 0.2) or (0.2 <= oflow < 1.0):
        return "normal"
    # Default to normal if not matched
    return "normal"

def main():
    if len(sys.argv) < 3:
        print("Usage: python runner1.py <video_path> <output_directory>")
        sys.exit(1)

    video_path = sys.argv[1].strip('"')
    frames_dir = sys.argv[2].strip('"')

    if not os.path.isfile(video_path):
        print(f"File not found: {video_path}")
        sys.exit(1)

    import time
    print(f"frames_dir: {repr(frames_dir)}")
    os.makedirs(frames_dir, exist_ok=True)
    start_time = time.time()
    print(f"Starting extraction at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

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
        writer.writerow(["Frame_i", "Frame_(i+1)", "PixelDiff", "SSIM", "HistDiff", "OpticalFlow", "Type"])

        ret, prev_frame = cap.read()
        if not ret:
            print("Failed to read first frame")
            sys.exit(1)

        frame_idx = 0
        print(f"Extracting frame {frame_idx}")
        cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_idx:06d}.png"), prev_frame)

        # Store indices for slow/identical frames
        flagged_indices = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            print(f"Extracting frame {frame_idx}")
            cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_idx:06d}.png"), frame)

            # Compute metrics between consecutive frames
            pdiff, ssim_val, hdiff, oflow = compute_metrics(prev_frame, frame)
            frame_type = classify_frame(pdiff, ssim_val, hdiff, oflow)
            writer.writerow([frame_idx - 1, frame_idx, pdiff, ssim_val, hdiff, oflow, frame_type])

            # Collect indices for slow/identical frames
            if frame_type in ["identical", "very_little_motion"]:
                flagged_indices.append((frame_idx - 1, frame_idx))

            prev_frame = frame

        print(f"Flagged frame pairs for interpolation: {flagged_indices}")
        # TODO: Call your interpolation function here, passing flagged_indices

    cap.release()
    end_time = time.time()
    print(f"Ending extraction at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    duration = end_time - start_time
    print(f"Total extraction duration: {duration:.2f} seconds ({duration/60:.2f} min)")
    print(f"Frames + metrics extracted to '{frames_dir}'")

if __name__ == "__main__":
    main()
