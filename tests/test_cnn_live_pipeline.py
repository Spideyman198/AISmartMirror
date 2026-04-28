"""Tests for CNN live pipeline helpers."""

import numpy as np

from vision.cnn_live_pipeline import (
    CNNLiveConfig,
    extract_crop,
    laplacian_blur_variance,
    sort_detections_left_to_right,
)


def test_extract_crop_margin_clamped():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = extract_crop(frame, (40, 40, 20, 20), margin_frac=0.5)
    assert crop.shape[0] > 0 and crop.shape[1] > 0


def test_laplacian_blur_variance_flat_image_low():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    v = laplacian_blur_variance(frame)
    assert v >= 0.0


def test_cnn_live_config_defaults():
    c = CNNLiveConfig()
    assert c.inference_interval >= 1
    assert c.confirmation_count >= 1


def test_sort_detections_left_to_right():
    dets = [
        {"bbox": (200, 0, 50, 50), "confidence": 0.9},
        {"bbox": (10, 0, 50, 50), "confidence": 0.9},
    ]
    s = sort_detections_left_to_right(dets)
    assert s[0]["bbox"][0] == 10
    assert s[1]["bbox"][0] == 200
