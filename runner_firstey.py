import os
import sys
import cv2
import numpy as np
import torch
import torchvision
import csv

def pixel_diff(frame1, frame2):
    """Mean absolute difference between two frames"""
    t1 = torch.from_numpy(frame1).float().to('cuda')
    t2 = torch.from_numpy(frame2).float().to('cuda')
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
        # Try GPU first
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
    t1 = torch.from_numpy(frame1).float().to('cuda') / 255.0
    t2 = torch.from_numpy(frame2).float().to('cuda') / 255.0
    t1 = t1.permute(2, 0, 1).unsqueeze(0)
    t2 = t2.permute(2, 0, 1).unsqueeze(0)

    pdiff = pixel_diff(frame1, frame2)
    try:
        ssim_val = torchvision.metrics.structural_similarity_index_measure(t1, t2).item()
    except Exception:
        ssim_val = float('nan')
    hdiff = histogram_diff(frame1, frame2)
    oflow = optical_flow_farnebäck(frame1, frame2)

    return pdiff, ssim_val, hdiff, oflow

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
        writer.writerow(["Frame_i", "Frame_(i+1)", "PixelDiff", "SSIM", "HistDiff", "OpticalFlow"])

        ret, prev_frame = cap.read()
        if not ret:
            print("Failed to read first frame")
            sys.exit(1)

        frame_idx = 0
        print(f"Extracting frame {frame_idx}")
        cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_idx:06d}.png"), prev_frame)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            print(f"Extracting frame {frame_idx}")
            cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_idx:06d}.png"), frame)

            # Compute metrics between consecutive frames
            pdiff, ssim_val, hdiff, oflow = compute_metrics(prev_frame, frame)
            writer.writerow([frame_idx - 1, frame_idx, pdiff, ssim_val, hdiff, oflow])

            # Debug log
           # print(f"Frame {frame_idx-1}-{frame_idx} | PixelDiff={pdiff:.4f}, SSIM={ssim_val:.4f}, HistDiff={hdiff:.4f}, OpticalFlow={oflow:.4f}")

            prev_frame = frame

    cap.release()
    end_time = time.time()
    print(f"Ending extraction at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    duration = end_time - start_time
    print(f"Total extraction duration: {duration:.2f} seconds ({duration/60:.2f} min)")
    print(f"Frames + metrics extracted to '{frames_dir}'")

if __name__ == "__main__":
    main()
