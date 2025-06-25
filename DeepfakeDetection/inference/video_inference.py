import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image

# 🔄 Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights)
num_features = model.classifier[1].in_features
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.4),
    torch.nn.Linear(num_features, 2)
)
model.load_state_dict(torch.load("models/best_model.pt", map_location=device))
model = model.to(device)
model.eval()

# 📦 Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 🎥 Extract N frames
def extract_frames(video_path, num_frames=10):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indexes = np.linspace(0, total - 1, num=num_frames, dtype=int)
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indexes:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames

# 🔍 Predict
def predict_video(video_path):
    frames = extract_frames(video_path)
    all_probs = []
    with torch.no_grad():
        for frame in frames:
            input_tensor = transform(frame).unsqueeze(0).to(device)
            out = model(input_tensor)
            prob = torch.softmax(out, dim=1)
            all_probs.append(prob.cpu())
    avg_prob = torch.mean(torch.stack(all_probs), dim=0)
    predicted = torch.argmax(avg_prob).item()
    return predicted, avg_prob.numpy()

# 🚀 Run on folder
video_folder = "videos_to_predict"
for vid in os.listdir(video_folder):
    if vid.endswith(".mp4"):
        path = os.path.join(video_folder, vid)
        label, prob = predict_video(path)
        print(f"{vid}: {'FAKE' if label == 1 else 'REAL'} | Real: {prob[0]:.3f}, Fake: {prob[1]:.3f}")
