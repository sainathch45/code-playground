import cv2
import os
import shutil
import random

def extract_and_split_videos(source_dir, dest_dir, split_ratio=0.8, frames_per_video=5, every_n_frames=15):
    for label in ['real', 'fake']:
        input_path = os.path.join(source_dir, label)
        video_files = [f for f in os.listdir(input_path) if f.endswith(".mp4")]
        random.shuffle(video_files)

        split_idx = int(len(video_files) * split_ratio)
        train_videos = video_files[:split_idx]
        val_videos = video_files[split_idx:]

        for subset, vids in [('train', train_videos), ('validation', val_videos)]:
            out_dir = os.path.join(dest_dir, subset, label)
            os.makedirs(out_dir, exist_ok=True)

            for vid in vids:
                cap = cv2.VideoCapture(os.path.join(input_path, vid))
                count = 0
                saved = 0
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                while saved < frames_per_video and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if count % every_n_frames == 0:
                        fname = f"{os.path.splitext(vid)[0]}_f{count}.jpg"
                        cv2.imwrite(os.path.join(out_dir, fname), frame)
                        saved += 1
                    count += 1
                cap.release()

        print(f"{label.upper()} videos split — Train: {len(train_videos)} | Val: {len(val_videos)}")

# 🧪 Example usage:
source = "videos/raw"
dest = "videos"
extract_and_split_videos(source, dest)
