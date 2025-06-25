import os
from torch.utils.data import Dataset
from PIL import Image

class HybridDeepfakeDataset(Dataset):
    def __init__(self, sources, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        class_map = {'real': 0, 'fake': 1}

        for path, override_label in sources:
            if override_label is None:
                for label in ['real', 'fake']:
                    subdir = os.path.join(path, label)
                    if os.path.isdir(subdir):
                        for fname in os.listdir(subdir):
                            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                                self.image_paths.append(os.path.join(subdir, fname))
                                self.labels.append(class_map[label])
            else:
                for fname in os.listdir(path):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(path, fname))
                        self.labels.append(override_label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
