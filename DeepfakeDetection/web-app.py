import gradio as gr
import torch
import mimetypes
from PIL import Image
import cv2
from torchvision.models import efficientnet_b0
from torchvision import transforms

# === Load Model ===
def load_model():
    model = efficientnet_b0()
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load("models/best_model-v3.pt", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# === Preprocessing ===
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Inference Logic ===
def predict_file(file_obj):
    if file_obj is None:
        return "⚠️ No file selected", "", None

    path = file_obj.name
    mime, _ = mimetypes.guess_type(path)

    if mime and mime.startswith("image"):
        img = Image.open(path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0]
            conf, pred = torch.max(probs, dim=0)
        label = "🟢 Real" if pred.item() == 0 else "🔴 Deepfake"
        return label, f"{conf.item()*100:.2f}%", img

    elif mime and mime.startswith("video"):
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return "❌ Error reading video", "", None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0]
            conf, pred = torch.max(probs, dim=0)
        label = "🟢 Real (1st frame)" if pred.item() == 0 else "🔴 Deepfake (1st frame)"
        return label, f"{conf.item()*100:.2f}%", img

    else:
        return "Unsupported file type", "", None

# === Gradio UI ===
with gr.Blocks(title="Deepfake Detector") as demo:
    gr.Markdown("## 🧠 Deepfake Detector\nDrop in an image or video below to analyze authenticity.")

    file_input = gr.File(
        label="Drop File Here",
        file_types=[".jpg", ".jpeg", ".png", ".mp4", ".mov"],
    )
    

    with gr.Row():
        prediction = gr.Textbox(label="Prediction", interactive=False)
        confidence = gr.Textbox(label="Confidence (%)", interactive=False)

    preview = gr.Image(label="Preview", interactive=False)

    def handle_input(file_obj):
        return predict_file(file_obj)

    file_input.change(
        fn=handle_input,
        inputs=file_input,
        outputs=[prediction, confidence, preview]
    )

demo.launch()
