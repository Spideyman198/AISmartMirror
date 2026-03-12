"""
Face recognizer - identifies known faces from face crops using local embeddings.

Uses face_recognition (dlib) for 128-dim embeddings. All processing is local;
no biometric data is sent to the cloud.

Threshold tradeoff:
- Lower threshold (e.g. 0.5): Stricter matching. Fewer false positives (wrong person
  recognized), but more false negatives (known person not recognized, especially
  at non-frontal angles).
- Higher threshold (e.g. 0.7): Lenient matching. Better recognition at angles, but
  higher risk of false positives. Tune via FACE_RECOGNITION_THRESHOLD in .env.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import face_recognition
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RecognitionResult:
    """Result of face recognition."""

    is_known: bool
    user_id: Optional[str] = None
    name: Optional[str] = None
    distance: float = 0.0  # Lower = more similar
    similarity: float = 0.0  # 0-1, higher = more similar (1 - normalized distance)
    best_match_name: Optional[str] = None  # Closest user even when rejected (for debug)


class FaceRecognizer:
    """
    Recognizes faces by matching embeddings against locally stored enrolled users.

    Supports multiple embeddings per user (from different angles) for better
    pose robustness. Uses Euclidean distance for matching.
    """

    def __init__(
        self,
        profiles_dir: Optional[Path] = None,
        threshold: float = 0.6,
    ) -> None:
        """
        Initialize the face recognizer.

        Args:
            profiles_dir: Path to directory with profiles.json and user embeddings.
            threshold: Max Euclidean distance for a match. Lower = stricter
                (fewer false matches, more misses). Higher = lenient (better at
                angles, more false matches). Default 0.6. Tune via .env.
        """
        self._profiles_dir = Path(profiles_dir) if profiles_dir else self._default_profiles_dir()
        self._threshold = threshold
        self._users: list[dict] = []
        self._embeddings: list[np.ndarray] = []
        self._load_profiles()
        logger.info(
            "Face recognizer initialized: %d users, threshold=%.2f",
            len(self._users),
            self._threshold,
        )

    def _default_profiles_dir(self) -> Path:
        """Default profiles directory relative to project root."""
        return Path(__file__).resolve().parent.parent / "data" / "known_faces"

    def _load_profiles(self) -> None:
        """Load user profiles and embeddings from disk.

        Supports both single embedding (128,) and multi-embedding (N, 128) per user.
        Multi-embedding improves recognition at non-frontal angles.
        """
        import json

        profiles_path = self._profiles_dir / "profiles.json"
        if not profiles_path.exists():
            logger.info("No profiles.json found at %s", profiles_path)
            return

        try:
            with open(profiles_path) as f:
                data = json.load(f)
            users = data.get("users", [])
            self._users = []
            self._embeddings = []  # List of arrays: each is (1, 128) or (N, 128)

            for u in users:
                user_id = u.get("user_id")
                embedding_file = u.get("embedding_file")
                if not user_id or not embedding_file:
                    continue
                emb_path = self._profiles_dir / embedding_file
                if not emb_path.exists():
                    logger.warning("Embedding file not found: %s", emb_path)
                    continue
                emb = np.load(emb_path)
                if emb.ndim == 1:
                    emb = emb.reshape(1, -1)
                # Keep as (N, 128) - no averaging; match against all
                self._users.append(u)
                self._embeddings.append(emb)

        except Exception as e:
            logger.exception("Failed to load profiles: %s", e)

    def _preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """Convert BGR to RGB for face_recognition."""
        if face_image is None or face_image.size == 0:
            return face_image
        if len(face_image.shape) == 2:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
        elif face_image.shape[2] == 3:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        return face_image

    def _encode(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Generate 128-dim embedding for a face crop."""
        rgb = self._preprocess(face_image)
        encodings = face_recognition.face_encodings(rgb)
        if not encodings:
            return None
        return encodings[0]

    def recognize(self, face_image: np.ndarray) -> RecognitionResult:
        """
        Identify a face from an image crop.

        Compares the live embedding against all stored embeddings for each user
        and returns the best match (lowest distance). If best distance <= threshold,
        the face is known; otherwise unknown.

        Args:
            face_image: BGR crop of a single face (from detector bbox).

        Returns:
            RecognitionResult with is_known, user_id, name, distance, similarity.
        """
        if face_image is None or face_image.size == 0:
            logger.debug("Recognition skipped: empty face image")
            return RecognitionResult(is_known=False)

        encoding = self._encode(face_image)
        if encoding is None:
            logger.debug("Recognition skipped: could not extract embedding (face landmarks?)")
            return RecognitionResult(is_known=False)

        if not self._embeddings:
            logger.debug("Recognition skipped: no enrolled users")
            return RecognitionResult(is_known=False, distance=0.0, similarity=0.0)

        # Flatten: one row per embedding, track user index per embedding
        all_embeddings = [e for user_embs in self._embeddings for e in user_embs]
        user_indices = [i for i, ue in enumerate(self._embeddings) for _ in range(len(ue))]

        # Euclidean distance (face_recognition uses this)
        distances = face_recognition.face_distance(all_embeddings, encoding)
        best_idx = int(np.argmin(distances))
        best_distance = float(distances[best_idx])
        user_idx = user_indices[best_idx]
        user = self._users[user_idx]

        # Convert distance to similarity: 0-1 scale (distance ~0-1.0 typical)
        similarity = max(0.0, 1.0 - best_distance)

        if best_distance <= self._threshold:
            logger.debug(
                "Recognition accepted: name=%s distance=%.3f threshold=%.2f",
                user.get("name"),
                best_distance,
                self._threshold,
            )
            return RecognitionResult(
                is_known=True,
                user_id=user.get("user_id"),
                name=user.get("name", user.get("user_id")),
                distance=best_distance,
                similarity=similarity,
            )

        best_name = user.get("name", user.get("user_id"))
        logger.debug(
            "Recognition rejected: best_match=%s distance=%.3f threshold=%.2f (distance > threshold)",
            best_name,
            best_distance,
            self._threshold,
        )
        return RecognitionResult(
            is_known=False,
            distance=best_distance,
            similarity=similarity,
            best_match_name=best_name,
        )

    def close(self) -> None:
        """Release resources."""
        self._users = []
        self._embeddings = []
        logger.debug("Face recognizer closed")
