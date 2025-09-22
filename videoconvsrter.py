import cv2
import os

image_folder = 'extracted_frames'
video_name = 'output_video.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort()

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
frame_size = (width, height)

out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 30, frame_size)

for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    out.write(frame)

out.release()

print(f"Video saved as {video_name}")