#!/usr/bin/env python3
"""
Live webcam test for the trained CNN face recognizer.

Uses vision.cnn_live_pipeline (smoothing, confirmation, throttled CNN, optional QC).

Usage:
    python scripts/test_cnn_live.py
    python scripts/test_cnn_live.py --debug --save-crops data/debug_live_crops

Defaults match tuned live settings (threshold / margin / interval / confirm).
Override any flag to experiment, or set CNN_CONFIDENCE_THRESHOLD / CNN_MIN_CLASS_MARGIN in .env for app integration later.
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import cv2

from config import get_settings
from utils.logger import setup_logging
from vision.camera_manager import CameraManager, CAP_DSHOW, CAP_MSMF
from vision.cnn_face_recognizer import CNNFaceRecognizer
from vision.cnn_live_pipeline import CNNLiveConfig, CNNLivePipeline
from vision.display import draw_face_boxes
from vision.face_detector import FaceDetector

# Pipeline defaults when flags omitted (interval / confirm not in .env yet)
DEFAULT_INTERVAL = 3
DEFAULT_CONFIRM = 5


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Live CNN face recognition test")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=project_root / "data" / "cnn_models",
        help="Directory with cnn_face_model.pt and class_mapping.json",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=settings.CNN_CONFIDENCE_THRESHOLD,
        help=f"Min softmax confidence for known (default {settings.CNN_CONFIDENCE_THRESHOLD} from .env / settings)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=settings.CNN_MIN_CLASS_MARGIN,
        help=f"Min top1-top2 softmax gap (default {settings.CNN_MIN_CLASS_MARGIN}; 0=off)",
    )
    parser.add_argument(
        "--max-entropy",
        type=float,
        default=None,
        help="Reject if softmax entropy exceeds this (nats). ~1.79=max confusion for 6 classes; try 1.35-1.5",
    )
    parser.add_argument(
        "--confirm",
        type=int,
        default=DEFAULT_CONFIRM,
        help=f"Consecutive matching raw labels before display updates (default {DEFAULT_CONFIRM})",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help=f"Run CNN every N frames (default {DEFAULT_INTERVAL})",
    )
    parser.add_argument(
        "--min-crop",
        type=int,
        default=48,
        help="Min crop width/height in pixels (default 48)",
    )
    parser.add_argument(
        "--min-blur",
        type=float,
        default=0.0,
        help="Min Laplacian variance (0=off; try 30-80 on your setup)",
    )
    parser.add_argument(
        "--min-det",
        type=float,
        default=0.0,
        help="Min face detector confidence (0=off; try 0.45-0.55)",
    )
    parser.add_argument(
        "--crop-margin",
        type=float,
        default=0.0,
        help="Expand bbox by this fraction before crop (0=match training; try 0.08)",
    )
    parser.add_argument(
        "--no-qc-detail",
        action="store_true",
        help="Hide quality-reject reasons in labels",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Log model path, mapping, crop/tensor shapes, top-3 softmax (set LOG_LEVEL=INFO)",
    )
    parser.add_argument(
        "--save-crops",
        type=Path,
        default=None,
        help="Save BGR face crops here for comparison with training data",
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="Disable left-to-right face ordering (for debugging only)",
    )
    args = parser.parse_args()

    if args.debug:
        setup_logging(level="INFO")

    model_pt = args.model_dir / "cnn_face_model.pt"
    if not model_pt.exists():
        print(f"No model at {model_pt}. Train first: python scripts/train_cnn_recognizer.py")
        sys.exit(1)

    recognizer = CNNFaceRecognizer(
        model_dir=args.model_dir,
        confidence_threshold=args.threshold,
        min_class_margin=args.margin,
        max_softmax_entropy=args.max_entropy,
        debug=args.debug,
    )

    cfg = CNNLiveConfig(
        inference_interval=max(1, args.interval),
        confirmation_count=max(1, args.confirm),
        min_crop_side=max(16, args.min_crop),
        min_blur_variance=max(0.0, args.min_blur),
        min_detector_confidence=max(0.0, args.min_det),
        crop_margin_frac=max(0.0, args.crop_margin),
        show_reject_detail=not args.no_qc_detail,
        stable_sort_faces=not args.no_sort,
        debug_recognize=args.debug,
        save_crops_dir=args.save_crops,
    )
    pipeline = CNNLivePipeline(recognizer=recognizer, config=cfg)

    mgr = CameraManager.find_working_camera(indexes=(0, 1, 2), backends=(CAP_DSHOW, CAP_MSMF))
    if mgr is None:
        print("Error: Could not open camera")
        sys.exit(1)

    detector = FaceDetector(min_confidence=0.5, model_selection=0)

    cv2.namedWindow("CNN live test", cv2.WINDOW_NORMAL)
    print("CNN live test. Q = quit.")
    print(
        f"  interval={cfg.inference_interval}  confirm={cfg.confirmation_count}  "
        f"threshold={args.threshold}  margin={args.margin}  max_entropy={args.max_entropy}"
    )
    if args.save_crops:
        args.save_crops.mkdir(parents=True, exist_ok=True)
        print(f"  Saving crops to {args.save_crops.resolve()}")

    try:
        while True:
            frame = mgr.read()
            if frame is None:
                continue

            detections = detector.detect(frame)
            labels, dets_draw = pipeline.process_frame(
                frame, detections, threshold_for_debug=args.threshold
            )

            display = frame.copy()
            draw_face_boxes(
                display,
                dets_draw if dets_draw else detections,
                labels=labels if labels else None,
            )
            cv2.imshow("CNN live test", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        mgr.close()
        detector.close()
        recognizer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
