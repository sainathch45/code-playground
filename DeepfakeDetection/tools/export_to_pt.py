import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from lightning_modules.detector import DeepfakeDetector

ckpt_path = "/project/models/best_model-v2.ckpt"
pt_output = "/project/models/best_model-v2.pt"

# Rebuild the same backbone as during training
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
backbone = efficientnet_b0(weights=weights)
num_features = backbone.classifier[1].in_features
backbone.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.4),
    torch.nn.Linear(num_features, 2)
)

# Now load LightningModule with that model
model = DeepfakeDetector.load_from_checkpoint(
    ckpt_path,
    model=backbone,
    lr=1e-4  # or whatever you used during training
)

# Save just the PyTorch model weights
torch.save(model.model.state_dict(), pt_output)

print(f"✅ Exported PyTorch model to {pt_output}")
