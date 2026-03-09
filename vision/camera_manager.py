"""
Camera manager - handles webcam initialization and frame capture.

Provides a clean interface for opening the camera, reading frames,
and releasing resources. Supports Windows backends (DSHOW, MSMF) and
Raspberry Pi / USB webcams.
"""

from typing import Optional

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

# OpenCV backend constants (Windows)
CAP_DSHOW = getattr(cv2, "CAP_DSHOW", 700)
CAP_MSMF = getattr(cv2, "CAP_MSMF", 1400)

BACKEND_NAMES = {
    CAP_DSHOW: "CAP_DSHOW",
    CAP_MSMF: "CAP_MSMF",
}


class CameraManager:
    """
    Manages webcam lifecycle and frame capture.

    Uses OpenCV VideoCapture. On Windows, supports explicit backend selection
    (CAP_DSHOW, CAP_MSMF) for more reliable capture.
    """

    def __init__(
        self,
        index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        backend: Optional[int] = None,
    ) -> None:
        """
        Initialize camera manager (does not open camera yet).

        Args:
            index: Camera device index (0 = default webcam).
            width: Frame width in pixels.
            height: Frame height in pixels.
            fps: Target frames per second.
            backend: OpenCV backend (CAP_DSHOW, CAP_MSMF), or None for default.
        """
        self._index = index
        self._width = width
        self._height = height
        self._fps = fps
        self._backend = backend
        self._cap: Optional[cv2.VideoCapture] = None
        self._backend_used: Optional[int] = None

    def open(self) -> bool:
        """
        Open the camera. On Windows, tries backend if specified, else default.

        Returns:
            True if camera opened and warm-up succeeded, False otherwise.
        """
        if self._backend is not None:
            return self._open_with_backend(self._index, self._backend)
        # Default: try default backend first
        cap = cv2.VideoCapture(self._index)
        if self._try_use_cap(cap, "default"):
            return True
        cap.release()
        return False

    def _open_with_backend(self, index: int, backend: int) -> bool:
        """Open camera with specific backend."""
        backend_name = BACKEND_NAMES.get(backend, f"backend_{backend}")
        logger.info("Opening camera index=%d backend=%s", index, backend_name)

        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            logger.error("Camera index=%d backend=%s: isOpened()=False", index, backend_name)
            return False

        logger.info("Camera index=%d backend=%s: isOpened()=True", index, backend_name)
        self._cap = cap
        self._backend_used = backend

        # Apply resolution/FPS
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        # Reduce buffer to 1 frame (avoids stale/black frames on some Windows cameras)
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("Camera index=%d backend=%s: resolution %dx%d", index, backend_name, actual_w, actual_h)

        # Warm-up: discard first few frames (often invalid on Windows)
        if not self._warm_up():
            logger.error("Camera warm-up failed: read() returned None")
            self.close()
            return False

        logger.info("Camera index=%d backend=%s: ready", index, backend_name)
        return True

    def _try_use_cap(self, cap: cv2.VideoCapture, backend_name: str) -> bool:
        """Try to use an already-created VideoCapture. Sets self._cap and runs warm-up."""
        if not cap.isOpened():
            logger.error("Camera index=%d backend=%s: isOpened()=False", self._index, backend_name)
            return False

        self._cap = cap
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if not self._warm_up():
            self._cap = None
            return False

        logger.info("Camera index=%d backend=%s: ready", self._index, backend_name)
        return True

    def _warm_up(self, num_frames: int = 20, min_success: int = 5) -> bool:
        """
        Read and discard initial frames. Many cameras return black/invalid frames at startup.

        Args:
            num_frames: Max frames to try.
            min_success: Require this many consecutive successful reads.

        Returns:
            True if we got min_success consecutive good frames.
        """
        consecutive_ok = 0
        for i in range(num_frames):
            ret, frame = self._cap.read()
            if ret and frame is not None and frame.size > 0:
                consecutive_ok += 1
                if consecutive_ok >= min_success:
                    logger.debug("Warm-up: %d consecutive good frames", min_success)
                    return True
            else:
                consecutive_ok = 0
        return False

    def grab(self) -> bool:
        """Grab next frame without decoding (advances buffer). Returns True if successful."""
        if self._cap is None or not self._cap.isOpened():
            return False
        return self._cap.grab()

    def flush_buffer(self, frames: int = 30) -> None:
        """Grab and discard frames to flush stale buffer."""
        for _ in range(frames):
            self.grab()

    def read(self) -> Optional[np.ndarray]:
        """
        Read the next frame from the camera.

        Returns:
            BGR frame as numpy array, or None if read failed.
        """
        if self._cap is None or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None
        return frame

    def close(self) -> None:
        """Release camera resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._backend_used = None
            logger.info("Camera released")

    def get_backend_name(self) -> str:
        """Return the backend used (for logging)."""
        if self._backend_used is None:
            return "default"
        return BACKEND_NAMES.get(self._backend_used, f"backend_{self._backend_used}")

    @staticmethod
    def try_open(index: int, backend: Optional[int] = None) -> Optional["CameraManager"]:
        """
        Try to open a camera at the given index with optional backend.

        Returns:
            CameraManager if successful, None otherwise.
        """
        mgr = CameraManager(index=index, backend=backend)
        if mgr.open():
            return mgr
        return None

    @staticmethod
    def find_working_camera(
        indexes: tuple[int, ...] = (0, 1, 2),
        backends: Optional[tuple[int, ...]] = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> Optional["CameraManager"]:
        """
        Try multiple index/backend combinations until one works.

        Args:
            indexes: Camera indexes to try.
            backends: Backends to try per index. If None, uses (CAP_DSHOW, CAP_MSMF) on Windows.
            width: Frame width.
            height: Frame height.
            fps: Target FPS.

        Returns:
            CameraManager if a working camera found, None otherwise.
        """
        if backends is None:
            backends = (CAP_DSHOW, CAP_MSMF)

        for index in indexes:
            for backend in backends:
                mgr = CameraManager(index=index, width=width, height=height, fps=fps, backend=backend)
                if mgr.open():
                    return mgr
        # Fallback: try default (no explicit backend)
        for index in indexes:
            mgr = CameraManager(index=index, width=width, height=height, fps=fps, backend=None)
            if mgr.open():
                return mgr
        return None
