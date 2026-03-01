"""
🔞 NSFW Detector Processor
Analyzes video frames for NSFW content and fires events.
Uses NudeDetector from nudenet (install: pip install nudenet).
Falls back to a heuristic skin-tone detector if nudenet isn't available.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# How often to run NSFW inference per participant (seconds)
INFERENCE_INTERVAL = 1.5
# Minimum confidence to trigger a warning
NSFW_THRESHOLD = 0.65

# NudeDetector labels considered NSFW
NSFW_LABELS = {
    "EXPOSED_ANUS",
    "EXPOSED_BREAST_F",
    "EXPOSED_BUTTOCKS",
    "EXPOSED_GENITALIA_F",
    "EXPOSED_GENITALIA_M",
    "EXPOSED_BREAST_M",
}


@dataclass
class NSFWEvent:
    type: str = "nsfw_detected"
    participant_id: str = ""
    confidence: float = 0.0
    category: str = ""


class NSFWDetectorProcessor:
    """
    Vision Agents processor that checks each participant's frames for NSFW content.
    Fires 'nsfw_detected' events that the agent subscribes to.
    """

    def __init__(self):
        self._running = True
        self._last_check: dict[str, float] = {}
        self._detector = self._load_detector()
        self._event_callbacks: list = []

    def _load_detector(self) -> tuple:
        """Try to load NudeDetector; fall back to heuristic detector."""
        try:
            from nudenet import NudeDetector
            detector = NudeDetector()
            logger.info("✅ NudeDetector loaded")
            return ("nudenet", detector)
        except ImportError:
            logger.warning(
                "⚠️  nudenet not installed — using heuristic skin detector. "
                "Install with: pip install nudenet"
            )
            return ("heuristic", None)
        except Exception as e:
            logger.warning(f"⚠️  NudeDetector failed to load ({e}) — using heuristic.")
            return ("heuristic", None)

    # ── Vision Agents processor interface ───────────────────────────────────

    async def process_frame(
        self,
        frame: np.ndarray,
        participant_id: str,
        emit_event=None,
    ) -> np.ndarray:
        """
        Called by Vision Agents per frame. Returns frame unmodified (detection only).
        Fires nsfw_detected event via emit_event callback.
        """
        if not self._running:
            return frame

        now = time.time()
        last = self._last_check.get(participant_id, 0)
        if now - last < INFERENCE_INTERVAL:
            return frame  # Rate limit inference

        self._last_check[participant_id] = now

        result = await asyncio.get_event_loop().run_in_executor(
            None, self._run_inference, frame
        )

        if result and result["confidence"] >= NSFW_THRESHOLD:
            logger.warning(
                f"🔞 NSFW frame detected: {participant_id} | "
                f"{result['category']} @ {result['confidence']:.0%}"
            )
            event = NSFWEvent(
                participant_id=participant_id,
                confidence=result["confidence"],
                category=result["category"],
            )
            if emit_event:
                await emit_event(event)
            for cb in self._event_callbacks:
                try:
                    await cb(event)
                except Exception as e:
                    logger.error(f"Event callback error: {e}")

        return frame

    def _run_inference(self, frame: np.ndarray) -> Optional[dict]:
        """Synchronous inference — runs in thread pool."""
        method, detector = self._detector
        if method == "nudenet":
            return self._nudenet_inference(frame, detector)
        return self._heuristic_inference(frame)

    def _nudenet_inference(self, frame: np.ndarray, detector) -> Optional[dict]:
        """Use NudeDetector for NSFW detection."""
        try:
            import tempfile, os
            # NudeDetector expects a file path
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                cv2.imwrite(f.name, frame)
                tmp_path = f.name

            detections = detector.detect(tmp_path)
            os.unlink(tmp_path)

            # detections: list of dicts with 'class' and 'score' keys
            # e.g. [{"class": "EXPOSED_BREAST_F", "score": 0.91, "box": [...]}]
            nsfw_hits = [
                d for d in detections
                if d.get("class") in NSFW_LABELS and d.get("score", 0) >= NSFW_THRESHOLD
            ]

            if not nsfw_hits:
                return None

            # Return the highest confidence hit
            top = max(nsfw_hits, key=lambda d: d["score"])
            return {
                "confidence": top["score"],
                "category": top["class"].replace("_", " ").title(),
            }

        except Exception as e:
            logger.debug(f"NudeDetector inference error: {e}")
            return None

    def _heuristic_inference(self, frame: np.ndarray) -> Optional[dict]:
        """
        Fallback: skin-tone pixel ratio heuristic.
        Conservative threshold to reduce false positives.
        """
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0, 40, 80], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_skin, upper_skin)

            skin_ratio = np.sum(mask > 0) / mask.size

            if skin_ratio > 0.55:
                confidence = min(0.95, skin_ratio * 1.4)
                return {
                    "confidence": confidence,
                    "category": "High Skin Exposure (heuristic)",
                }
            return None
        except Exception as e:
            logger.debug(f"Heuristic inference error: {e}")
            return None

    def register_event_callback(self, callback):
        self._event_callbacks.append(callback)

    async def stop_processing(self):
        self._running = False

    async def close(self):
        self._running = False
        logger.info("NSFWDetectorProcessor closed.")