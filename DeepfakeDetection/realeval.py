import os
import torch
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.models import efficientnet_b0

# === Load model ===
model = efficientnet_b0()
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load("models/best_model-v2.pt", map_location="cpu"))
model.eval()

# === Preprocessing with optional noise ===
def distort(image, simulate=True):
    if simulate:
        image = image.resize((224, 224))
        arr = np.array(image).astype(np.uint8)

        if np.random.rand() < 0.5:
            arr = cv2.GaussianBlur(arr, (5, 5), 0)

        if np.random.rand() < 0.5:
            _, arr = cv2.imencode('.jpg', arr, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
            arr = cv2.imdecode(arr, 1)

        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(arr)

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])(image).unsqueeze(0)

# === Run on test folder ===
def evaluate(folder="realworld_samples/", simulate_noise=True):
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if not os.path.isfile(path):
            continue
        try:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image = Image.open(path).convert("RGB")
                tensor = distort(image, simulate=simulate_noise)
            elif file.lower().endswith((".mp4", ".mov")):
                cap = cv2.VideoCapture(path)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    print(f"{file}: ❌ Error reading video")
                    continue
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                tensor = distort(image, simulate=simulate_noise)
            else:
                print(f"{file}: Unsupported type")
                continue

            with torch.no_grad():
                out = model(tensor)
                prob = torch.softmax(out, dim=1)[0]
                conf, pred = torch.max(prob, dim=0)

            label = "Real" if pred.item() == 0 else "Deepfake"
            print(f"{file:<30} ➤ {label:<9} ({conf.item()*100:.2f}%)")

        except Exception as e:
            print(f"{file}: ⚠️ {e}")

# Example use
if __name__ == "__main__":
    evaluate(simulate_noise=True)
