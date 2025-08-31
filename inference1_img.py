import os
import glob
import cv2
import torch
import time
from model.RIFE import Model   # ✅ adjust if your RIFE model path is different

def read_image(path, device):
    """Read and preprocess an image for RIFE"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    return img

def check_model_weights():
    """Check if the model weights exist, and download them if they don't"""
    model_path = os.path.join("train_log", "flownet.pkl")
    if not os.path.isfile(model_path):
        print("Model weights not found. Attempting to download...")
        model_dir = os.path.dirname(model_path)
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            import urllib.request
            url = "https://github.com/hzwer/Practical-RIFE/releases/download/1.0/flownet.pkl"
            print(f"Downloading from {url}...")
            urllib.request.urlretrieve(url, model_path)
            print("Download complete!")
        except Exception as e:
            print(f"Failed to download model weights: {str(e)}")
            print("Please download the model weights manually and place them in train_log/flownet.pkl")
            print("Download URL: https://github.com/hzwer/Practical-RIFE/releases/download/1.0/flownet.pkl")
            return False
    return True

def interpolate_folder(input_folder, output_folder, start_index=0):
    os.makedirs(output_folder, exist_ok=True)

    # Collect all images
    files = sorted(
        glob.glob(os.path.join(input_folder, '*.png')) +
        glob.glob(os.path.join(input_folder, '*.jpg')) +
        glob.glob(os.path.join(input_folder, '*.jpeg'))
    )

    print(f"Found {len(files)} images in {input_folder}")
    if len(files) < 2:
        print("Not enough images to interpolate (need at least 2).")
        return

    # Check if model weights exist and download if needed
    if not check_model_weights():
        print("Cannot proceed without model weights.")
        return

    # Load RIFE model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU for inference (this will be slow).")
        print("For better performance, install CUDA drivers and PyTorch with CUDA support.")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        
    try:
        model = Model()
        model.load_model("train_log", -1)
        model.eval()
        model.device()
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Try running setup.py to install dependencies and download model weights.")
        return

    out_idx = start_index
    total_pairs = len(files) - 1

    # Timer start
    total_start = time.time()

    for i in range(total_pairs):
        pair_start = time.time()

        img1_path = files[i]
        img2_path = files[i + 1]

        print(f"\n▶ Processing pair {i+1}/{total_pairs}: {os.path.basename(img1_path)} + {os.path.basename(img2_path)}")

        # Save parent frame (img1)
        parent1 = cv2.imread(img1_path)
        out_name_parent = f"img{out_idx:04d}.png"
        cv2.imwrite(os.path.join(output_folder, out_name_parent), parent1)
        out_idx += 1

        # Read both frames for interpolation
        img1 = read_image(img1_path, device)
        img2 = read_image(img2_path, device)

        # Interpolate
        with torch.no_grad():
            mid = model.inference(img1, img2)

        # Convert tensor → numpy image
        mid_img = (mid[0].cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
        mid_img = cv2.cvtColor(mid_img, cv2.COLOR_RGB2BGR)

        # Save interpolated frame
        out_name_interp = f"img{out_idx:04d}.png"
        cv2.imwrite(os.path.join(output_folder, out_name_interp), mid_img)
        out_idx += 1

        pair_end = time.time()
        print(f"   ⏱ Time for this pair: {pair_end - pair_start:.3f} sec")

    # Save the very last parent frame
    last_parent = cv2.imread(files[-1])
    out_name_last = f"img{out_idx:04d}.png"
    cv2.imwrite(os.path.join(output_folder, out_name_last), last_parent)

    # Timer end
    total_end = time.time()
    print(f"\n✅ Saved {out_idx} frames (parents + interpolated)")
    print(f"⏱ Total time: {total_end - total_start:.2f} sec")
    print(f"⚡ Avg time per pair: {(total_end - total_start)/total_pairs:.3f} sec")
    print(f"Output saved to: {output_folder}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Interpolate frames in a folder")
    parser.add_argument("--input", type=str, default="extracted_frame", 
                        help="Input folder containing frames to interpolate")
    parser.add_argument("--output", type=str, default="output", 
                        help="Output folder for interpolated frames")
    args = parser.parse_args()
    
    # Convert relative paths to absolute paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.input):
        args.input = os.path.join(script_dir, args.input)
    if not os.path.isabs(args.output):
        args.output = os.path.join(script_dir, args.output)
    
    print(f"Input folder: {args.input}")
    print(f"Output folder: {args.output}")
    
    interpolate_folder(args.input, args.output)
