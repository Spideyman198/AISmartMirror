#!/usr/bin/env python3
"""
Health check script - verifies the AISmartMirror environment is ready.

Run from project root: python scripts/health_check.py
Use --skip-camera to bypass camera check (e.g. in CI or without webcam).
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Track failures for exit code
FAILED = False


def log_ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def log_fail(msg: str) -> None:
    global FAILED
    FAILED = True
    print(f"  [FAIL] {msg}")


def log_skip(msg: str) -> None:
    print(f"  [SKIP] {msg}")


def check_python() -> bool:
    """Verify Python version."""
    print("\n1. Python environment")
    if sys.version_info >= (3, 10):
        log_ok(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        return True
    log_fail(f"Python 3.10+ required, got {sys.version_info.major}.{sys.version_info.minor}")
    return False


def check_imports() -> bool:
    """Verify required packages import correctly."""
    print("\n2. Required packages")
    packages = [
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("mediapipe", "mediapipe"),
        ("dotenv", "python-dotenv"),
    ]
    for module, pkg in packages:
        try:
            __import__(module)
            log_ok(f"{pkg}")
        except ImportError as e:
            log_fail(f"{pkg}: {e}")
            return False
    return True


def check_camera(skip: bool) -> bool:
    """Verify camera can be initialized."""
    print("\n3. Camera")
    if skip:
        log_skip("Camera check skipped (--skip-camera)")
        return True
    try:
        from vision.camera_manager import CameraManager

        manager = CameraManager(index=0)
        if manager.open():
            frame = manager.read()
            manager.close()
            if frame is not None:
                log_ok(f"Camera opened, frame shape {frame.shape}")
                return True
            log_fail("Camera opened but read() returned None")
        else:
            log_fail("Could not open camera (no device or permission denied)")
    except Exception as e:
        log_fail(f"Camera error: {e}")
    return False


def check_app_entry_point() -> bool:
    """Verify app entry point can be loaded."""
    print("\n4. App entry point")
    try:
        from app.main import main
        from app.app_controller import AppController

        if not callable(main):
            log_fail("main is not callable")
            return False
        controller = AppController()
        if controller is None:
            log_fail("AppController() returned None")
            return False
        log_ok("App entry point loads correctly")
        return True
    except Exception as e:
        log_fail(f"App import error: {e}")
        return False


def main() -> int:
    skip_camera = "--skip-camera" in sys.argv
    print("AISmartMirror Health Check")
    print("=" * 40)

    check_python()
    check_imports()
    check_camera(skip=skip_camera)
    check_app_entry_point()

    print("\n" + "=" * 40)
    if FAILED:
        print("Health check FAILED. Fix the issues above.")
        return 1
    print("Health check PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
