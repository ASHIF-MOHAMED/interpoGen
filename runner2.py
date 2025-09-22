import os
import sys
import cv2
import ffmpeg
import glob
import time
from multiprocessing import Pool
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import numpy as np
from PIL import Image

def extract_frames(video_path, frames_dir):
    if not os.path.isfile(video_path):
        print(f"File not found: {video_path}")
        sys.exit(1)

    # Read video properties
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(" Could not open the video file.")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0
    cap.release()

    print(f"🎥 Input Video: {video_path}")
    print(f"📊 FPS: {fps}")
    print(f"🖼 Total Frames: {total_frames}")
    print(f"⏱ Duration: {duration_sec:.2f} sec ({duration_sec/60:.2f} min)")

    os.makedirs(frames_dir, exist_ok=True)

    print(f"📤 Extracting frames to: {frames_dir} ...")
    try:
        (
            ffmpeg
            .input(video_path)
            .output(os.path.join(frames_dir, 'frame_%06d.png'), start_number=0)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print(" FFmpeg error:", e.stderr.decode())
        sys.exit(1)

    print(f"✅ Done! Extracted frames saved to: {frames_dir}")

def init_model(model_name='RealESRGAN_x2plus'):
    base_dir = r"E:\mini project\ECCV2022-RIFE\Real-ESRGAN\weights"
    model_path = os.path.join(base_dir, model_name + '.pth')

    if model_name == 'RealESRGAN_x2plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=2)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    upsampler = RealESRGANer(
        scale=2,
        model_path=model_path,
        model=model,
        tile=300,
        tile_pad=10,
        pre_pad=0,
        half=True
    )
    return upsampler


def process_chunk(image_list, output_folder, model_name):
    upsampler = init_model(model_name)
    for img_path in image_list:
        try:
            img_name = os.path.basename(img_path)
            print(f"[PID {os.getpid()}] Processing {img_name}")

            img = Image.open(img_path).convert('RGB')
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            output, _ = upsampler.enhance(img, outscale=2)

            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            output_img = Image.fromarray(output)
            output_img.save(os.path.join(output_folder, img_name))

        except Exception as e:
            print(f"Error processing {img_path}: {e}")


def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python script.py <input_video> <extracted_frames_dir> <upscaled_frames_dir>")
        sys.exit(1)

    video_path = sys.argv[1]
    extracted_frames_dir = sys.argv[2]
    upscaled_frames_dir = sys.argv[3]
    model_name = 'RealESRGAN_x2plus'
    num_parts = 3

    # Step 1: Extract frames
    extract_frames(video_path, extracted_frames_dir)

    # Step 2: Enhance resolution
    os.makedirs(upscaled_frames_dir, exist_ok=True)
    images = sorted(glob.glob(os.path.join(extracted_frames_dir, '*')))
    start_time = time.time()
    chunks = chunkify(images, num_parts)

    with Pool(num_parts) as pool:
        pool.starmap(process_chunk, [(chunk, upscaled_frames_dir, model_name) for chunk in chunks])

    print(f"Resolution enhancement finished in {time.time() - start_time:.2f} seconds")
