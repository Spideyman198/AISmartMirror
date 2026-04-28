#!/usr/bin/env python3
"""
Collect face crops for CNN training dataset.

Saves detected face crops to data/cnn_faces/<user_id>/ for each user.
Run for each user, then run prepare_cnn_dataset.py to split into train/val.

Usage:
    python scripts/collect_cnn_faces.py --name alice
    python scripts/collect_cnn_faces.py --name bob --target 50
    python scripts/collect_cnn_faces.py --name bob --extra 50   # add 50 more (on top of existing count)

--target = stop when total files in folder reaches this number. If you already have more than
that, the script does nothing (use --extra or a higher --target).

Press SPACE to capture. Q to quit.
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import cv2

from vision.camera_manager import CameraManager, CAP_DSHOW, CAP_MSMF
from vision.face_detector import FaceDetector

CNN_FACES_DIR = project_root / "data" / "cnn_faces"
MIN_FACE_SIZE = 64


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect face crops for CNN training")
    parser.add_argument("--name", "-n", required=True, help="User ID (e.g. alice)")
    parser.add_argument("--target", "-t", type=int, default=100,
                        help="Stop when folder has this many images total (default 100)")
    parser.add_argument(
        "--extra",
        type=int,
        default=None,
        help="Add this many new images on top of current count (overrides --target if set)",
    )
    parser.add_argument("--output", "-o", default=None, help="Override output directory")
    args = parser.parse_args()

    user_id = args.name.lower().replace(" ", "_")
    out_dir = Path(args.output) if args.output else CNN_FACES_DIR / user_id
    out_dir.mkdir(parents=True, exist_ok=True)

    count = len(list(out_dir.glob("*.jpg"))) + len(list(out_dir.glob("*.png")))

    if args.extra is not None:
        target_total = count + max(0, args.extra)
    else:
        target_total = args.target

    if count >= target_total:
        print(f"Collecting for {user_id}. Current: {count}. Target total: {target_total}.")
        print("Nothing to collect — you already have enough images for this target.")
        print("  Use a higher --target, or e.g. --extra 50 to add 50 more images.")
        sys.exit(0)

    mgr = CameraManager.find_working_camera(indexes=(0, 1, 2), backends=(CAP_DSHOW, CAP_MSMF))
    if mgr is None:
        print("Error: Could not open camera")
        sys.exit(1)

    detector = FaceDetector(min_confidence=0.5, model_selection=0)

    print(f"Collecting for {user_id}. Target total: {target_total}. Current: {count}")
    print("SPACE=capture, Q=quit")
    print()

    cv2.namedWindow("Collect CNN Faces", cv2.WINDOW_NORMAL)

    while count < target_total:
        frame = mgr.read()
        if frame is None:
            continue

        detections = detector.detect(frame)
        display = frame.copy()

        if len(detections) == 1:
            x, y, w, h = detections[0]["bbox"]
            if w >= MIN_FACE_SIZE and h >= MIN_FACE_SIZE:
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    display, f"SPACE to capture ({count}/{target_total})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
            else:
                cv2.putText(display, "Face too small - move closer", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif len(detections) > 1:
            cv2.putText(display, "Only one face please", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Collect CNN Faces", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord(" ") and len(detections) == 1:
            x, y, w, h = detections[0]["bbox"]
            if w >= MIN_FACE_SIZE and h >= MIN_FACE_SIZE:
                crop = frame[y : y + h, x : x + w]
                count += 1
                path = out_dir / f"face_{count:04d}.jpg"
                cv2.imwrite(str(path), crop)
                print(f"  Saved {count}/{target_total}: {path.name}")

    mgr.close()
    detector.close()
    cv2.destroyAllWindows()
    print(f"\nDone. Total images in folder: {count} -> {out_dir}")


if __name__ == "__main__":
    main()
