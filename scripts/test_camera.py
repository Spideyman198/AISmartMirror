#!/usr/bin/env python3
"""
Standalone camera test script.

Tries camera indexes 0, 1, 2 with CAP_DSHOW and CAP_MSMF backends.
Displays webcam if successful. Press 'q' to quit.

Usage: python scripts/test_camera.py
       python scripts/test_camera.py --backends DSHOW MSMF
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import cv2

from vision.camera_manager import (
    CameraManager,
    CAP_DSHOW,
    CAP_MSMF,
    BACKEND_NAMES,
)


def main() -> None:
    print("AISmartMirror - Camera Test")
    print("=" * 50)

    backends = (CAP_DSHOW, CAP_MSMF)
    indexes = (0, 1, 2)

    for index in indexes:
        for backend in backends:
            backend_name = BACKEND_NAMES.get(backend, f"backend_{backend}")
            print(f"\nTrying index={index} backend={backend_name}...")

            mgr = CameraManager(index=index, backend=backend)
            if mgr.open():
                print(f"\n*** SUCCESS: index={index} backend={backend_name} ***")
                print(f"isOpened()=True, warm-up read() succeeded")
                print("Displaying webcam. Press 'q' to quit.")
                print("=" * 50)

                cv2.namedWindow("Camera Test", cv2.WINDOW_NORMAL)
                while True:
                    frame = mgr.read()
                    if frame is None:
                        print("Frame read failed during display")
                        break
                    cv2.imshow("Camera Test", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                mgr.close()
                cv2.destroyAllWindows()
                return

            mgr.close()

    # Fallback: try default backend
    print("\nTrying default backend...")
    for index in indexes:
        print(f"  index={index}...")
        mgr = CameraManager(index=index, backend=None)
        if mgr.open():
            print(f"\n*** SUCCESS: index={index} backend=default ***")
            print("Displaying webcam. Press 'q' to quit.")
            print("=" * 50)

            cv2.namedWindow("Camera Test", cv2.WINDOW_NORMAL)
            while True:
                frame = mgr.read()
                if frame is None:
                    print("Frame read failed during display")
                    break
                cv2.imshow("Camera Test", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            mgr.close()
            cv2.destroyAllWindows()
            return

        mgr.close()

    print("\n*** FAILED: No working camera found ***")
    print("Tried indexes", indexes, "with backends CAP_DSHOW, CAP_MSMF, and default")
    sys.exit(1)


if __name__ == "__main__":
    main()
