import cv2
import os

def extract_frames_from_video(video_path, out_dir, every_n_frames=15):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame_count % every_n_frames == 0:
            out_path = os.path.join(out_dir, f"{video_name}_{saved_count}.jpg")
            cv2.imwrite(out_path, frame)
            saved_count += 1
        frame_count += 1
    cap.release()

# Example usage
video_dir = "source_videos/fake"
output_dir = "dataset/train/fake"
os.makedirs(output_dir, exist_ok=True)

for video in os.listdir(video_dir):
    if video.endswith(".mp4"):
        extract_frames_from_video(os.path.join(video_dir, video), output_dir)
