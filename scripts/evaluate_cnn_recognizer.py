#!/usr/bin/env python3
"""
Evaluate trained CNN face recognizer on validation set.

Usage:
    python scripts/evaluate_cnn_recognizer.py
    python scripts/evaluate_cnn_recognizer.py --model-dir data/cnn_models
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from vision.cnn_face_dataset import CNNFaceDataset, DEFAULT_INPUT_SIZE

CNN_FACES_DIR = project_root / "data" / "cnn_faces"
CNN_MODELS_DIR = project_root / "data" / "cnn_models"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CNN face recognizer")
    parser.add_argument("--data-dir", type=Path, default=CNN_FACES_DIR)
    parser.add_argument("--model-dir", type=Path, default=CNN_MODELS_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    model_path = args.model_dir / "cnn_face_model.pt"
    mapping_path = args.model_dir / "class_mapping.json"

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Run train_cnn_recognizer.py first.")
        sys.exit(1)

    val_ds = CNNFaceDataset(args.data_dir, split="val", input_size=DEFAULT_INPUT_SIZE)
    if len(val_ds) == 0:
        print("Error: No validation data.")
        sys.exit(1)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Load model (derive num_classes from saved state for consistency)
    from vision.cnn_face_model import create_model
    state = torch.load(model_path, map_location="cpu")
    num_classes = state["classifier.1.weight"].shape[0]
    model = create_model(num_classes, pretrained=False)
    model.load_state_dict(state)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    correct = 0
    total = 0
    class_correct = {}
    class_total = {}

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(labels.size(0)):
                c = labels[i].item()
                class_total[c] = class_total.get(c, 0) + 1
                if predicted[i] == labels[i]:
                    class_correct[c] = class_correct.get(c, 0) + 1

    overall_acc = 100.0 * correct / total
    print(f"Overall accuracy: {correct}/{total} = {overall_acc:.1f}%")
    print("\nPer-class accuracy:")
    # Prefer class_mapping.json for display (matches model); fallback to val_ds
    if mapping_path.exists():
        with open(mapping_path) as f:
            mapping = json.load(f)
        idx_to_class = {int(k): v for k, v in mapping["idx_to_class"].items()}
    else:
        idx_to_class = val_ds.idx_to_class
    for idx in sorted(class_total.keys()):
        acc = 100.0 * class_correct.get(idx, 0) / class_total[idx]
        print(f"  {idx_to_class[idx]}: {class_correct.get(idx, 0)}/{class_total[idx]} = {acc:.1f}%")


if __name__ == "__main__":
    main()
