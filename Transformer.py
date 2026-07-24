# import os
# import cv2
# from ultralytics import YOLO
#
# # 1. Load your existing model
# model = YOLO(r'C:\Users\Jason\PycharmProjects\PythonProject\train2model.pt')
#
# # 2. Define directories
# images_dir = 'extracted_frames/'  # Where your 2,000 new images are
# labels_dir = 'extracted_labels/'  # Where the .txt files will go
# os.makedirs(labels_dir, exist_ok=True)
#
# # 3. Loop through all images and auto-label
# for filename in os.listdir(images_dir):
#     if filename.endswith(('.jpg', '.png')):
#         img_path = os.path.join(images_dir, filename)
#
#         # Run inference (set confidence fairly high so it doesn't guess)
#         results = model(img_path, conf=0.40, verbose=False)[0]
#
#         # Create a text file for this image
#         txt_filename = os.path.splitext(filename)[0] + '.txt'
#         txt_path = os.path.join(labels_dir, txt_filename)
#
#         with open(txt_path, 'w') as f:
#             for box in results.boxes:
#                 # YOLO format: class x_center y_center width height (normalized)
#                 cls_id = int(box.cls[0].item())
#                 x_c, y_c, w, h = box.xywhn[0].tolist()
#
#                 # Write to file
#                 f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
#
# print("Auto-labeling complete. Ready for manual review!")

import cv2
import os

# --- Configuration ---
VIDEO_PATH = r'highlight_reel.mp4'  # Update this to your video's exact path
OUTPUT_DIR = r'extracted_frames'  # The folder where images will be saved

# Set how many frames you want to skip.
# If the video is 60 FPS:
# EXTRACT_EVERY_N_FRAMES = 30 means it saves 2 images per second.
# EXTRACT_EVERY_N_FRAMES = 15 means it saves 4 images per second.
EXTRACT_EVERY_N_FRAMES = 10


def extract_frames(video_path, output_dir, skip_frames):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}. Check the file path.")
        return

    # Get video properties for the progress readout
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Video Loaded: {total_frames} total frames at {fps} FPS.")
    print(f"Extraction Target: Saving 1 frame every {skip_frames} frames.\n")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        # Break the loop if the video has ended
        if not ret:
            break

        # Check if the current frame is one we want to save
        if frame_count % skip_frames == 0:
            filename = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

            # Print progress every 100 saved frames
            if saved_count % 100 == 0:
                print(f"Progress: Saved {saved_count} images so far...")

        frame_count += 1

    # Cleanup
    cap.release()
    print("-" * 40)
    print(f"SUCCESS: Extraction complete!")
    print(f"Total images saved to '{output_dir}': {saved_count}")


if __name__ == '__main__':
    extract_frames(VIDEO_PATH, OUTPUT_DIR, EXTRACT_EVERY_N_FRAMES)