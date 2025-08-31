import os
import subprocess
import glob
import shutil

def interpolate_folder(input_folder, output_folder, inference_script):
    os.makedirs(output_folder, exist_ok=True)

    images = sorted(
        glob.glob(os.path.join(input_folder, '*.png')) +
        glob.glob(os.path.join(input_folder, '*.jpg')) +
        glob.glob(os.path.join(input_folder, '*.jpeg'))
    )

    print(f"Found {len(images)} images in {input_folder}")
    if len(images) < 2:
        print("Not enough images to interpolate (need at least 2).")
        return

    for i in range(len(images) - 1):
        img1 = images[i]
        img2 = images[i + 1]

        base1 = os.path.splitext(os.path.basename(img1))[0]
        base2 = os.path.splitext(os.path.basename(img2))[0]
        out_name = f'interpolated_{base1}_{base2}.png'
        out_path = os.path.join(output_folder, out_name)

        # clear old outputs first
        if os.path.exists("output"):
            shutil.rmtree("output")

        # Always assign cmd before using it
        cmd = [
            'python', inference_script,
            '--img', img1, img2,
            '--exp', '1'
        ]
        print(f'▶ Interpolating: {img1} + {img2} -> {out_path}')
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"❌ Error running inference_img.py: {result.stderr}")
        else:
            # grab the middle interpolated frame from "output/"
            generated = sorted(glob.glob("output/*.png"))
            if len(generated) >= 3:
                # img0, middle, img1 -> take the middle one
                middle_frame = generated[1]
                shutil.move(middle_frame, out_path)
                print(f"✅ Saved: {out_path}")
            else:
                print("⚠ No interpolated frame found in output/")

if __name__ == '__main__':
    input_folder = r'E:\mini project\ECCV2022-RIFE\frames'
    output_folder = r'E:\mini project\ECCV2022-RIFE\frames\interpolated_frames'
    inference_script = r'E:\mini project\ECCV2022-RIFE\inference_img.py'
    interpolate_folder(input_folder, output_folder, inference_script)
