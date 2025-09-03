import os
import glob
import cv2
import torch
import time
from model.RIFE import Model   # ✅ adjust if your RIFE model path is different

def read_image(path, device):
    """Read and preprocess an image for RIFE with dimension padding"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    
    # Get original dimensions
    h, w = img.shape[:2]
    
    # Pad dimensions to be divisible by 32 (RIFE requirement)
    def pad_to_multiple(x, multiple=32):
        return ((x + multiple - 1) // multiple) * multiple
    
    new_h = pad_to_multiple(h, 32)
    new_w = pad_to_multiple(w, 32)
    
    # Pad image if needed
    if new_h != h or new_w != w:
        # Calculate padding
        pad_h = new_h - h
        pad_w = new_w - w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    return img, (h, w)  # Return original dimensions for cropping later

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

def interpolate_folder(input_folder, output_folder, start_index=0, selective_pairs=None):
    """
    Interpolate frames in a folder.
    Args:
        input_folder: Path to input images
        output_folder: Path to save interpolated frames
        start_index: Starting index for output naming
        selective_pairs: List of (i, j) tuples for selective interpolation. If None, interpolates all pairs.
    """
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
    
    # Determine which pairs to process
    if selective_pairs is not None:
        pairs_to_process = selective_pairs
        total_pairs = len(selective_pairs)
        print(f"Selective interpolation: Processing {total_pairs} flagged pairs")
    else:
        pairs_to_process = [(i, i+1) for i in range(len(files) - 1)]
        total_pairs = len(files) - 1
        print(f"Full interpolation: Processing {total_pairs} consecutive pairs")

    # Timer start
    total_start = time.time()

    for idx, (i, j) in enumerate(pairs_to_process):
        pair_start = time.time()

        img1_path = files[i]
        img2_path = files[j]

        print(f"\n▶ Processing pair {idx+1}/{total_pairs}: {os.path.basename(img1_path)} + {os.path.basename(img2_path)}")

        # Save parent frame (img1)
        parent1 = cv2.imread(img1_path)
        out_name_parent = f"img{out_idx:04d}.png"
        cv2.imwrite(os.path.join(output_folder, out_name_parent), parent1)
        out_idx += 1

        # Read both frames for interpolation
        img1, orig_dims1 = read_image(img1_path, device)
        img2, orig_dims2 = read_image(img2_path, device)

        # Interpolate
        with torch.no_grad():
            mid = model.inference(img1, img2)

        # Convert tensor → numpy image and crop back to original size
        mid_img = (mid[0].cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
        mid_img = cv2.cvtColor(mid_img, cv2.COLOR_RGB2BGR)
        
        # Crop back to original dimensions
        orig_h, orig_w = orig_dims1  # Assuming both images have same original dimensions
        current_h, current_w = mid_img.shape[:2]
        
        # Calculate crop coordinates (center crop)
        start_h = (current_h - orig_h) // 2
        start_w = (current_w - orig_w) // 2
        mid_img = mid_img[start_h:start_h+orig_h, start_w:start_w+orig_w]

        # Save interpolated frame
        out_name_interp = f"img{out_idx:04d}.png"
        cv2.imwrite(os.path.join(output_folder, out_name_interp), mid_img)
        out_idx += 1

        pair_end = time.time()
        print(f"   ⏱ Time for this pair: {pair_end - pair_start:.3f} sec")

    # Save the very last parent frame (only if processing all pairs)
    if selective_pairs is None:
        last_parent = cv2.imread(files[-1])
        out_name_last = f"img{out_idx:04d}.png"
        cv2.imwrite(os.path.join(output_folder, out_name_last), last_parent)

    # Timer end
    total_end = time.time()
    print(f"\n✅ Saved {out_idx} frames (parents + interpolated)")
    print(f"⏱ Total time: {total_end - total_start:.2f} sec")
    print(f"⚡ Avg time per pair: {(total_end - total_start)/max(1, total_pairs):.3f} sec")
    print(f"Output saved to: {output_folder}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Interpolate frames in a folder")
    parser.add_argument("--input", type=str, default="extracted_frame", 
                        help="Input folder containing frames to interpolate")
    parser.add_argument("--output", type=str, default="output", 
                        help="Output folder for interpolated frames")
    parser.add_argument("--selective", type=str, default=None,
                        help="Path to CSV file with flagged pairs for selective interpolation")
    args = parser.parse_args()
    
    # Convert relative paths to absolute paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.input):
        args.input = os.path.join(script_dir, args.input)
    if not os.path.isabs(args.output):
        args.output = os.path.join(script_dir, args.output)
    
    print(f"Input folder: {args.input}")
    print(f"Output folder: {args.output}")
    
    # Load selective pairs if provided
    selective_pairs = None
    if args.selective and os.path.isfile(args.selective):
        import csv
        selective_pairs = []
        with open(args.selective, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get('Type') in ['identical', 'very_little_motion']:
                    frame_i = int(row['Frame_i'])
                    frame_j = int(row['Frame_(i+1)'])
                    selective_pairs.append((frame_i, frame_j))
        print(f"Loaded {len(selective_pairs)} flagged pairs for selective interpolation")
    
    interpolate_folder(args.input, args.output, selective_pairs=selective_pairs)
