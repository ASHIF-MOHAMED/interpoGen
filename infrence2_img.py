import os
import cv2
import torch
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

def get_model(model_dir):
    print("Loading IFNet model from IFNet_2R.py and weights from train_log/flownet.pkl...")
    from model.IFNet_2R import IFNet
    import torch
    model = IFNet()
    weights_path = os.path.join(model_dir, 'flownet.pkl')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    print("Model loaded successfully.")
    return model

def read_image(path, device):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = (torch.tensor(img.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    return img


def main(input_folder, output_folder, model_dir='train_log'):
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Model directory: {model_dir}")
    model = get_model(model_dir)

    # Supported image extensions
    exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    files = [f for f in os.listdir(input_folder) if os.path.splitext(f)[1].lower() in exts]
    files.sort()  # Ensure correct order
    print(f"Found {len(files)} frames: {files}")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # Find the last used index in output folder
    existing_files = [f for f in os.listdir(output_folder) if f.endswith('.png') or f.endswith('.jpg')]
    if existing_files:
        existing_indices = [int(''.join(filter(str.isdigit, os.path.splitext(f)[0]))) for f in existing_files if any(c.isdigit() for c in os.path.splitext(f)[0])]
        start_index = max(existing_indices) + 1 if existing_indices else 0
        print(f"Existing output files: {existing_files}")
        print(f"Starting index for new frames: {start_index}")
    else:
        start_index = 0
        print("No existing output files. Starting from index 0.")

    out_idx = start_index
    for i in range(len(files) - 1):
        img1_path = os.path.join(input_folder, files[i])
        img2_path = os.path.join(input_folder, files[i+1])
        print(f"Interpolating pair {i+1}/{len(files)-1}: {files[i]} + {files[i+1]}")
        img1 = read_image(img1_path, device)
        img2 = read_image(img2_path, device)
        # Concatenate images along channel dimension (shape: [1, 6, H, W])
        input_tensor = torch.cat([img1, img2], dim=1)
        # IFNet expects input shape [batch, 6, H, W]
        with torch.no_grad():
            # The forward method returns several outputs; merged[2] is the final interpolated frame
            _, _, merged, _, _, _ = model(input_tensor)
            mid_frame = merged[2]
        # Save interpolated frame
        out_name = f"img{out_idx:04d}.png"
        cv2.imwrite(os.path.join(output_folder, out_name), (mid_frame[0] * 255).byte().cpu().numpy().transpose(1, 2, 0))
        print(f"Saved interpolated frame: {out_name}")
        out_idx += 1
    print(f"Done! Interpolated {len(files)-1} pairs.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Batch interpolation for folder of frames')
    parser.add_argument('--input', required=True, help='Input folder with frames')
    parser.add_argument('--output', required=True, help='Output folder for interpolated frames')
    parser.add_argument('--model', default='train_log', help='Model directory')
    args = parser.parse_args()
    main(args.input, args.output, args.model)
