"""
CNN face model - MobileNetV2 architecture for face recognition.

Shared between training, evaluation, and inference.
"""

def create_model(num_classes: int, pretrained: bool = True):
    """
    Create MobileNetV2 with custom classifier head.

    MobileNetV2: ~3.5M params, suitable for embedded/Raspberry Pi.
    Input: 224x224 RGB. Output: num_classes logits.
    """
    import torch.nn as nn
    from torchvision.models import mobilenet_v2

    model = mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(1280, num_classes),
    )
    return model
