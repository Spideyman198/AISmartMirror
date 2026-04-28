"""
CNN face recognizer - inference module for trained MobileNetV2 classifier.

Loads trained model and class mapping.

Input: BGR face crop (from detector). Output: RecognitionResult (class name or Unknown).
Can be plugged into the live app alongside the existing embedding-based recognizer.

Differs from face_recognizer (dlib):
- This: CNN classifier, outputs class label directly. Trained on your users.
- Dlib: Embedding + distance. No training; compares to stored embeddings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

# ImageNet normalization (must match training)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
DEFAULT_INPUT_SIZE = 224


@dataclass
class CNNRecognitionResult:
    """Result of CNN face recognition."""

    is_known: bool
    user_id: Optional[str] = None
    name: Optional[str] = None
    class_idx: int = -1
    confidence: float = 0.0  # Softmax probability of predicted class
    # top1 - top2; high when the winner is separated from the runner-up
    margin: float = 0.0
    reject_reason: Optional[str] = None  # e.g. "low_margin", "low_confidence", "high_entropy"
    # Shannon entropy of softmax (nats); high = spread across classes, low = peaked
    softmax_entropy: float = 0.0
    # Populated when recognize(..., debug=True)
    debug_top3: tuple[tuple[str, int, float], ...] = field(default_factory=tuple)


class CNNFaceRecognizer:
    """
    Recognizes faces using a trained MobileNetV2 classifier.

    Expects face crops from the detector. Resizes to 224x224, normalizes,
    runs inference. Returns class name (user_id) and confidence.
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        input_size: int = DEFAULT_INPUT_SIZE,
        confidence_threshold: float = 0.5,
        min_class_margin: float = 0.0,
        max_softmax_entropy: Optional[float] = None,
        debug: bool = False,
    ) -> None:
        """
        Args:
            model_dir: Path to data/cnn_models/ (contains cnn_face_model.pt, class_mapping.json)
            input_size: Model input size (224 for MobileNetV2)
            confidence_threshold: Min softmax prob to consider "known" (else Unknown)
            min_class_margin: If > 0, require (top1 - top2) softmax margin; else Unknown.
                Reduces ambiguous predictions between similar identities.
            max_softmax_entropy: If set, reject as unknown when entropy exceeds this (nats).
                Catches indecisive distributions. For 6 classes, uniform ≈ 1.79; try 1.35–1.55.
                Does not stop overconfident wrong IDs (use threshold + margin).
            debug: If True, log paths, mapping, crop/tensor shapes, top-3 softmax (via recognize(debug=True))
        """
        self._model_dir = Path(model_dir) if model_dir else self._default_model_dir()
        self._model_dir = self._model_dir.resolve()
        self._input_size = input_size
        self._confidence_threshold = confidence_threshold
        self._min_class_margin = min_class_margin
        self._max_softmax_entropy = max_softmax_entropy
        self._debug = debug

        self._model = None
        self._idx_to_class: dict[int, str] = {}
        self._class_to_idx: dict[str, int] = {}
        self._model_path: Optional[Path] = None
        self._mapping_path: Optional[Path] = None
        self._num_classes: int = 0
        self._load_model()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _default_model_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent / "data" / "cnn_models"

    def _load_model(self) -> None:
        """Load trained model and class mapping."""
        model_path = self._model_dir / "cnn_face_model.pt"
        mapping_path = self._model_dir / "class_mapping.json"
        self._model_path = model_path.resolve()
        self._mapping_path = mapping_path.resolve() if mapping_path.exists() else None

        if not model_path.exists():
            logger.info("CNN model not found at %s: CNN recognizer disabled", model_path)
            return

        try:
            import torch
            from vision.cnn_face_model import create_model

            state = torch.load(model_path, map_location="cpu")
            num_classes = state["classifier.1.weight"].shape[0]
            self._num_classes = num_classes

            model = create_model(num_classes, pretrained=False)
            model.load_state_dict(state)
            model.eval()
            self._model = model

            if mapping_path.exists():
                import json
                with open(mapping_path) as f:
                    mapping = json.load(f)
                self._idx_to_class = {int(k): v for k, v in mapping["idx_to_class"].items()}
                self._class_to_idx = mapping["class_to_idx"]
            else:
                logger.warning("No class_mapping.json next to model at %s", mapping_path)

            keys = sorted(self._idx_to_class.keys())
            if len(self._idx_to_class) != num_classes or keys != list(range(num_classes)):
                logger.warning(
                    "Class mapping may not match model: num_classes=%d, mapping keys=%s "
                    "(expected contiguous 0..%d). Wrong file pair causes misclassification.",
                    num_classes,
                    keys,
                    num_classes - 1,
                )

            logger.info(
                "CNN face recognizer loaded: %d classes from %s",
                len(self._idx_to_class),
                self._model_dir,
            )
            logger.info("CNN model weights: %s", self._model_path)
            if self._mapping_path:
                logger.info("CNN class_mapping: %s", self._mapping_path)
                logger.info("CNN idx_to_class: %s", dict(sorted(self._idx_to_class.items())))
            if self._debug:
                logger.info(
                    "CNN debug mode on: will log crop shape, tensor shape, top-3 softmax on recognize(debug=True)"
                )
        except Exception as e:
            logger.exception("Failed to load CNN model: %s", e)

    def _preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """Resize (INTER_LINEAR, matches torchvision default), normalize, CHW float."""
        if face_image is None or face_image.size == 0:
            return None
        rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb,
            (self._input_size, self._input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        normalized = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return normalized.transpose(2, 0, 1)  # HWC -> CHW

    def recognize(
        self,
        face_image: np.ndarray,
        *,
        debug: Optional[bool] = None,
        save_crop_path: Optional[Path] = None,
    ) -> CNNRecognitionResult:
        """
        Classify a face crop.

        Args:
            face_image: BGR crop of a single face (from detector bbox).
            debug: If True, log crop shape, tensor shape, top-3 classes. Defaults to self._debug.
            save_crop_path: If set, write the BGR crop to this path (for dataset comparison).

        Returns:
            CNNRecognitionResult with is_known, name, confidence.
        """
        dbg = self._debug if debug is None else debug
        empty = CNNRecognitionResult(is_known=False)

        if self._model is None:
            return empty

        if face_image is None or face_image.size == 0:
            return empty

        if save_crop_path is not None:
            save_crop_path = Path(save_crop_path)
            save_crop_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_crop_path), face_image)

        if dbg:
            logger.info(
                "CNN debug: crop shape=%s dtype=%s min/max=%.1f/%.1f",
                face_image.shape,
                face_image.dtype,
                float(face_image.min()),
                float(face_image.max()),
            )

        x = self._preprocess(face_image)
        if x is None:
            return empty

        try:
            import torch

            x_tensor = torch.from_numpy(x).float().unsqueeze(0)  # NCHW
            if dbg:
                logger.info(
                    "CNN debug: tensor shape=%s (N,C,H,W)",
                    tuple(x_tensor.shape),
                )

            with torch.no_grad():
                logits = self._model(x_tensor)
            probs = torch.softmax(logits, dim=1)
            n_cls = probs.shape[1]
            k3 = min(3, n_cls)
            top3v, top3i = probs[0].topk(k3)
            top3_tuples: list[tuple[str, int, float]] = []
            for j in range(k3):
                ci = int(top3i[j])
                pv = float(top3v[j])
                name = self._idx_to_class.get(ci, f"<missing idx {ci}>")
                top3_tuples.append((name, ci, pv))
            if dbg:
                logger.info("CNN debug: top-%d softmax: %s", k3, top3_tuples)

            if n_cls < 2:
                conf = float(probs[0].max())
                idx = int(probs[0].argmax())
                margin = 1.0
            else:
                topv, topi = probs[0].topk(2)
                conf = float(topv[0])
                idx = int(topi[0])
                margin = float(topv[0] - topv[1])

            # Entropy in nats: -sum(p log p)
            p = probs[0].clamp(min=1e-12)
            entropy = float(-(p * p.log()).sum())

            if self._max_softmax_entropy is not None and entropy > self._max_softmax_entropy:
                return CNNRecognitionResult(
                    is_known=False,
                    class_idx=idx,
                    confidence=conf,
                    margin=margin,
                    reject_reason="high_entropy",
                    softmax_entropy=entropy,
                    debug_top3=tuple(top3_tuples),
                )

            if conf < self._confidence_threshold:
                return CNNRecognitionResult(
                    is_known=False,
                    class_idx=idx,
                    confidence=conf,
                    margin=margin,
                    reject_reason="low_confidence",
                    softmax_entropy=entropy,
                    debug_top3=tuple(top3_tuples),
                )

            if self._min_class_margin > 0.0 and margin < self._min_class_margin:
                return CNNRecognitionResult(
                    is_known=False,
                    class_idx=idx,
                    confidence=conf,
                    margin=margin,
                    reject_reason="low_margin",
                    softmax_entropy=entropy,
                    debug_top3=tuple(top3_tuples),
                )

            if idx in self._idx_to_class:
                user_id = self._idx_to_class[idx]
                return CNNRecognitionResult(
                    is_known=True,
                    user_id=user_id,
                    name=user_id.replace("_", " ").title(),
                    class_idx=idx,
                    confidence=conf,
                    margin=margin,
                    softmax_entropy=entropy,
                    debug_top3=tuple(top3_tuples),
                )

            if dbg:
                logger.warning(
                    "CNN debug: predicted idx=%d not in idx_to_class (mapping/model mismatch?)",
                    idx,
                )
            return CNNRecognitionResult(
                is_known=False,
                class_idx=idx,
                confidence=conf,
                margin=margin,
                softmax_entropy=entropy,
                debug_top3=tuple(top3_tuples),
            )
        except Exception as e:
            logger.debug("CNN recognition failed: %s", e)
            return CNNRecognitionResult(is_known=False)

    def close(self) -> None:
        """Release resources."""
        self._model = None
        self._idx_to_class = {}
        self._class_to_idx = {}
        logger.debug("CNN face recognizer closed")
