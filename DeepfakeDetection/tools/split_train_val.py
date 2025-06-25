import os
import shutil
import random

def split_dataset(source_dir, dest_dir, split_ratio=0.8):
    for label in ['real', 'fake']:
        src = os.path.join(source_dir, label)
        all_files = [f for f in os.listdir(src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(all_files)

        split_idx = int(len(all_files) * split_ratio)
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]

        for category, files in [('train', train_files), ('validation', val_files)]:
            out_dir = os.path.join(dest_dir, category, label)
            os.makedirs(out_dir, exist_ok=True)
            for f in files:
                shutil.copy(os.path.join(src, f), os.path.join(out_dir, f))

        print(f"✅ Split {label} — Train: {len(train_files)}, Val: {len(val_files)}")

# 🧪 Example usage
source_dataset = "/home/you/data/celebdf/raw"  # This should contain `real/` and `fake/`
destination = "/home/you/data/celebdf"         # Will create `train/` and `validation/`

split_dataset(source_dataset, destination)
