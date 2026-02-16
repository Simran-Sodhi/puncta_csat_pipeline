#!/usr/bin/env python3
"""
spotiflow_detector.py — Spotiflow deep-learning puncta detector (optional).

Wraps the Spotiflow library for GPU-accelerated spot detection.
If Spotiflow is not installed the module imports cleanly but
:meth:`detect_2d` raises ``ImportError`` with installation instructions.

Install Spotiflow::

    pip install spotiflow
"""

import numpy as np
from .core import BaseDetector, PunctaDetectionResult

_SPOTIFLOW_AVAILABLE = False
try:
    from spotiflow.model import Spotiflow as _Spotiflow
    _SPOTIFLOW_AVAILABLE = True
except ImportError:
    _Spotiflow = None


def is_available() -> bool:
    """Return True if the spotiflow package is importable."""
    return _SPOTIFLOW_AVAILABLE


class SpotiflowDetector(BaseDetector):
    """Deep-learning spot detector based on Spotiflow.

    Parameters
    ----------
    model_path : str
        ``"general"`` for the pretrained general model, or a filesystem
        path to a fine-tuned checkpoint directory.
    device : str
        ``"auto"`` (use CUDA if available), ``"cpu"``, or ``"cuda"``.
    prob_threshold : float
        Minimum probability to accept a detection (0–1).
    min_distance : int
        Minimum distance (pixels) between detections.
    subpixel : bool
        Use sub-pixel localisation refinement.
    """

    def __init__(
        self,
        model_path: str = "general",
        device: str = "auto",
        prob_threshold: float = 0.5,
        min_distance: int = 2,
        subpixel: bool = True,
    ):
        self.model_path = model_path
        self.device = device
        self.prob_threshold = prob_threshold
        self.min_distance = min_distance
        self.subpixel = subpixel
        self._model = None

    def get_name(self) -> str:
        return "Spotiflow (deep learning)"

    def get_default_params(self) -> dict:
        return {
            "model_path": "general",
            "device": "auto",
            "prob_threshold": 0.5,
            "min_distance": 2,
            "subpixel": True,
        }

    # -------------------------------------------------------------- #
    #  Model loading (lazy)
    # -------------------------------------------------------------- #

    def _ensure_model(self):
        if self._model is not None:
            return
        if not _SPOTIFLOW_AVAILABLE:
            raise ImportError(
                "Spotiflow is not installed.  Install it with:\n"
                "    pip install spotiflow\n"
                "See https://github.com/weigertlab/spotiflow"
            )
        device = self.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.model_path == "general":
            self._model = _Spotiflow.from_pretrained("general")
        else:
            self._model = _Spotiflow.from_folder(self.model_path)

        # Move model to requested device after loading
        import torch
        self._model = self._model.to(torch.device(device))

    # -------------------------------------------------------------- #
    #  Detection
    # -------------------------------------------------------------- #

    def detect_2d(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
        **kwargs,
    ) -> PunctaDetectionResult:
        self._ensure_model()

        # Spotiflow expects float32 in [0, 1] or uint8/uint16
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / img.max()

        points, details = self._model.predict(
            img,
            prob_thresh=self.prob_threshold,
            min_distance=self.min_distance,
            subpix=self.subpixel,
        )
        # points: (N, 2) as [y, x]
        # details: object with .prob attribute
        if len(points) == 0:
            return PunctaDetectionResult.empty()

        coords = np.asarray(points, dtype=np.float64)
        probs = np.asarray(details.prob, dtype=np.float64) if hasattr(details, "prob") else np.ones(len(coords))

        # Filter by mask
        if mask is not None:
            keep = []
            for i, (y, x) in enumerate(coords):
                yi, xi = int(round(y)), int(round(x))
                if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1]:
                    if mask[yi, xi] > 0:
                        keep.append(i)
            if not keep:
                return PunctaDetectionResult.empty()
            coords = coords[keep]
            probs = probs[keep]

        return PunctaDetectionResult(
            coordinates=coords,
            confidences=probs,
            metadata={
                "detector": self.get_name(),
                "model": self.model_path,
                "prob_threshold": self.prob_threshold,
            },
        )

    # -------------------------------------------------------------- #
    #  Training / fine-tuning
    # -------------------------------------------------------------- #

    def fine_tune(
        self,
        train_images: list,
        train_coords: list,
        val_images: list | None = None,
        val_coords: list | None = None,
        epochs: int = 50,
        lr: float = 3e-4,
        save_path: str = "models/spotiflow_finetuned",
    ) -> dict:
        """Fine-tune Spotiflow on custom annotated data.

        Parameters
        ----------
        train_images : list of ndarray
            Training images (2-D float32).
        train_coords : list of ndarray
            Ground truth coordinates per image, each (N, 2) as [y, x].
        save_path : str
            Where to save the fine-tuned model checkpoint.

        Returns
        -------
        dict
            Training history.
        """
        if not _SPOTIFLOW_AVAILABLE:
            raise ImportError("Spotiflow is not installed.")

        from spotiflow.model import Spotiflow
        from spotiflow.utils import points_matching_dataset
        from pathlib import Path
        import os

        os.makedirs(save_path, exist_ok=True)

        # Build Spotiflow-compatible datasets
        # (This follows Spotiflow's training API)
        model = Spotiflow.from_pretrained("general")
        model.train(
            train_images, train_coords,
            val_images, val_coords,
            save_dir=save_path,
            epochs=epochs,
            learning_rate=lr,
        )
        return {"save_path": save_path, "epochs": epochs}
