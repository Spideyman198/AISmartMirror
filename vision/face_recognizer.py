"""
Face recognizer - identifies known faces from face crops.

Placeholder for future implementation. Will use face encodings
and a known-faces database (local, edge-first).
"""

from typing import Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class FaceRecognizer:
    """
    Recognizes faces by matching against a known-faces database.

    Placeholder: not implemented yet.
    """

    def __init__(self, known_faces_dir: Optional[str] = None) -> None:
        """
        Initialize the face recognizer.

        Args:
            known_faces_dir: Path to directory with known face images.
        """
        self._known_faces_dir = known_faces_dir
        logger.info("FaceRecognizer placeholder (not yet implemented)")

    def recognize(self, face_image: np.ndarray) -> Optional[str]:
        """
        Identify a face from an image crop.

        Args:
            face_image: RGB crop of a single face.

        Returns:
            Name of recognized person, or None if unknown.
        """
        # TODO: Implement with face-recognition or similar
        return None

    def close(self) -> None:
        """Release resources."""
        pass
