"""Vision module - camera, face detection, face recognition, gesture recognition."""

from vision.camera_manager import CameraManager
from vision.face_detector import FaceDetector
from vision.display import draw_face_boxes

__all__ = ["CameraManager", "FaceDetector", "draw_face_boxes"]
