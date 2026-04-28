"""
Live CNN recognition pipeline: quality gates, throttled inference, smoothing.

Keeps detection every frame; runs the CNN only every N frames and reuses the last
stable display labels between runs. Uses RecognitionSmoother so identities do not
flicker and switches require consecutive agreeing predictions.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from vision.cnn_face_recognizer import CNNFaceRecognizer, CNNRecognitionResult
from vision.face_recognizer import RecognitionResult
from vision.recognition_smoother import RecognitionSmoother


def laplacian_blur_variance(bgr: np.ndarray) -> float:
    """Higher = sharper. Typical faces: tens to hundreds depending on scale."""
    if bgr is None or bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def sort_detections_left_to_right(detections: list[dict]) -> list[dict]:
    """
    Stable left-to-right order by bbox center x.

    MediaPipe detection order can vary frame-to-frame. Without this, the
    RecognitionSmoother can attach the wrong identity to the wrong box when
    two faces are present.
    """
    if len(detections) <= 1:
        return detections

    def center_x(det: dict) -> float:
        x, y, w, h = det["bbox"]
        return float(x) + float(w) * 0.5

    return sorted(detections, key=center_x)


def extract_crop(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    margin_frac: float = 0.0,
) -> np.ndarray:
    """Crop face region; optional symmetric expansion (clamped to frame)."""
    x, y, w, h = bbox
    fh, fw = frame.shape[:2]
    if margin_frac > 0:
        dx = int(w * margin_frac * 0.5)
        dy = int(h * margin_frac * 0.5)
        x = max(0, x - dx)
        y = max(0, y - dy)
        w = min(fw - x, w + 2 * dx)
        h = min(fh - y, h + 2 * dy)
    return frame[y : y + h, x : x + w]


@dataclass
class CNNLiveConfig:
    """Tuning knobs for live CNN recognition."""

    inference_interval: int = 3
    """Run CNN every N frames (per tick). 1 = every frame (more lag)."""

    confirmation_count: int = 5
    """Consecutive matching raw labels before the displayed identity updates."""

    min_crop_side: int = 48
    """Reject crops smaller than this width or height (pixels)."""

    min_blur_variance: float = 0.0
    """If > 0, reject crops below this Laplacian variance (blur)."""

    min_detector_confidence: float = 0.0
    """If > 0, reject detections with MediaPipe score below this."""

    crop_margin_frac: float = 0.0
    """Expand bbox before crop. 0 matches tight training crops from collect_cnn_faces."""

    label_unknown: str = "Unknown"
    """Displayed name when not known."""

    show_reject_detail: bool = True
    """If True, show short reason for quality reject (tiny/blur/low det)."""

    stable_sort_faces: bool = True
    """Sort detections left-to-right before indexing (fixes multi-face label swap)."""

    debug_recognize: bool = False
    """Pass debug=True into CNN recognize() (logs top-3, shapes)."""

    save_crops_dir: Optional[Path] = None
    """If set, save each BGR crop here for comparison with training images."""


@dataclass
class CNNLivePipeline:
    """
    Stateful pipeline: frame counter, smoother, cached labels for skipped frames.
    """

    recognizer: CNNFaceRecognizer
    config: CNNLiveConfig = field(default_factory=CNNLiveConfig)
    _smoother: RecognitionSmoother = field(init=False, repr=False)
    _frame_count: int = field(default=0, init=False)
    _last_face_count: int = field(default=-1, init=False)
    _display_labels: list[str] = field(default_factory=list, init=False)
    _crop_save_seq: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        n = max(1, self.config.confirmation_count)
        self._smoother = RecognitionSmoother(confirmation_count=n)

    def reset(self) -> None:
        self._frame_count = 0
        self._last_face_count = -1
        self._display_labels = []
        self._smoother.update([], 0.0)

    def process_frame(
        self,
        frame: np.ndarray,
        detections: list[dict],
        threshold_for_debug: float = 0.5,
    ) -> tuple[list[str], list[dict]]:
        """
        Return (display strings, detections to draw) — same length, aligned.

        If stable_sort_faces is True, detections are sorted left-to-right; use the
        returned list for drawing so labels match boxes.

        Runs CNN only when the interval elapses or the number of faces changes.
        """
        self._frame_count += 1
        dets = list(detections)
        if self.config.stable_sort_faces:
            dets = sort_detections_left_to_right(dets)

        n = len(dets)
        interval = max(1, self.config.inference_interval)

        run_cnn = (
            self._frame_count % interval == 0
            or n != self._last_face_count
        )

        if not dets:
            self._display_labels = []
            self._last_face_count = 0
            self._smoother.update([], threshold_for_debug)
            return [], []

        if not run_cnn and len(self._display_labels) == n:
            return self._display_labels, dets

        self._last_face_count = n
        results: list[RecognitionResult] = []
        qc_notes: list[Optional[str]] = []

        for det in dets:
            r, note = self._process_one_face(frame, det)
            results.append(r)
            qc_notes.append(note)

        smooth_names, _ = self._smoother.update(results, threshold=threshold_for_debug)
        self._display_labels = []
        for i, lab in enumerate(smooth_names):
            sim = results[i].similarity
            base = lab if lab != "Unknown" else self.config.label_unknown
            note = qc_notes[i]
            if note and base == self.config.label_unknown:
                suffix = note if self.config.show_reject_detail else f"{sim:.2f}"
                self._display_labels.append(f"{base} ({suffix})")
            else:
                self._display_labels.append(f"{base} ({sim:.2f})")

        return self._display_labels, dets

    def _process_one_face(
        self, frame: np.ndarray, det: dict
    ) -> tuple[RecognitionResult, Optional[str]]:
        """Returns (result for smoother, optional QC note for display only)."""
        bbox = det["bbox"]
        det_score = float(det.get("confidence", 1.0))
        cfg = self.config

        if cfg.min_detector_confidence > 0.0 and det_score < cfg.min_detector_confidence:
            return self._reject_result("low det", 0.0)

        crop = extract_crop(frame, bbox, cfg.crop_margin_frac)
        if crop.size == 0:
            return self._reject_result("empty crop", 0.0)

        ch, cw = crop.shape[:2]
        if cw < cfg.min_crop_side or ch < cfg.min_crop_side:
            return self._reject_result(f"small {cw}x{ch}", 0.0)

        if cfg.min_blur_variance > 0.0:
            lap = laplacian_blur_variance(crop)
            if lap < cfg.min_blur_variance:
                return self._reject_result(f"blur {lap:.0f}", 0.0)

        save_path = None
        if cfg.save_crops_dir is not None:
            self._crop_save_seq += 1
            save_path = Path(cfg.save_crops_dir) / f"live_crop_{self._crop_save_seq:05d}.jpg"

        cnn: CNNRecognitionResult = self.recognizer.recognize(
            crop,
            debug=cfg.debug_recognize,
            save_crop_path=save_path,
        )
        sim = cnn.confidence
        if cnn.is_known and cnn.name:
            return (
                RecognitionResult(
                    is_known=True,
                    user_id=cnn.user_id,
                    name=cnn.name,
                    distance=1.0 - sim,
                    similarity=sim,
                    best_match_name=cnn.name,
                ),
                None,
            )
        return (
            RecognitionResult(
                is_known=False,
                name=None,
                distance=1.0 - sim,
                similarity=sim,
                best_match_name=cnn.name,
            ),
            None,
        )

    def _reject_result(
        self, reason: str, sim: float
    ) -> tuple[RecognitionResult, Optional[str]]:
        return (
            RecognitionResult(
                is_known=False,
                name=None,
                distance=1.0,
                similarity=sim,
            ),
            reason,
        )
