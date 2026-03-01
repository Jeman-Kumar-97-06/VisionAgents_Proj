"""
🎂 Age Detector Processor
Estimates participant age in real-time using DeepFace.
Fires two types of events:
  - age_detected        : informational, carries estimated age + bucket
  - minor_detected      : fires when estimated age < 18, triggers strict mode
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# How often to run age inference per participant (seconds)
# Age doesn't change — no need to run every frame
INFERENCE_INTERVAL = 8.0

# Age below which strict moderation kicks in
MINOR_AGE_THRESHOLD = 18

# Age buckets for clean reporting
AGE_BUCKETS = [
    (0,  12,  "child (under 12)"),
    (13, 17,  "teen (13–17)"),
    (18, 24,  "young adult (18–24)"),
    (25, 34,  "adult (25–34)"),
    (35, 49,  "adult (35–49)"),
    (50, 64,  "adult (50–64)"),
    (65, 120, "senior (65+)"),
]


def age_to_bucket(age: int) -> str:
    for lo, hi, label in AGE_BUCKETS:
        if lo <= age <= hi:
            return label
    return "unknown"


@dataclass
class AgeDetectedEvent:
    type: str = "age_detected"
    participant_id: str = ""
    estimated_age: int = 0
    age_bucket: str = ""
    is_minor: bool = False


@dataclass
class MinorDetectedEvent:
    type: str = "minor_detected"
    participant_id: str = ""
    estimated_age: int = 0
    age_bucket: str = ""


class AgeDetectorProcessor:
    """
    Vision Agents processor that estimates participant age from video frames.

    DeepFace is used as the primary model (runs on CPU, no GPU needed).
    Falls back to a lightweight OpenCV + heuristic approach if DeepFace
    isn't installed.

    Events fired (via emit_event callback):
        AgeDetectedEvent   — every successful detection
        MinorDetectedEvent — when estimated age < MINOR_AGE_THRESHOLD
    """

    def __init__(self):
        self._running = True
        self._last_check: dict[str, float] = {}
        # Cache last known age per participant to avoid repeat minor alerts
        self._last_age: dict[str, int] = {}
        # Track if we've already alerted about a minor (avoid spam)
        self._minor_alerted: set[str] = set()
        self._backend = self._load_backend()
        self._event_callbacks: list = []

    def _load_backend(self) -> tuple:
        """Try DeepFace first, fall back to heuristic."""
        try:
            from deepface import DeepFace
            # Warm up by running a tiny dummy analysis instead of build_model()
            # (DeepFace's build_model API changed — task/model_name args are internal now)
            logger.info("🔄 Warming up DeepFace age model...")
            import numpy as np
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            DeepFace.analyze(
                img_path=dummy,
                actions=["age"],
                enforce_detection=False,
                silent=True,
            )
            logger.info("✅ DeepFace age model ready")
            return ("deepface", DeepFace)
        except ImportError:
            logger.warning(
                "⚠️  DeepFace not installed — using heuristic age estimator. "
                "Install with: pip install deepface tf-keras"
            )
            return ("heuristic", None)
        except Exception as e:
            logger.warning(
                f"⚠️  DeepFace failed to initialize ({e}) — using heuristic age estimator."
            )
            return ("heuristic", None)

    # ── Vision Agents processor interface ───────────────────────────────────

    async def process_frame(
        self,
        frame: np.ndarray,
        participant_id: str,
        emit_event=None,
    ) -> np.ndarray:
        """Called by Vision Agents per frame. Returns frame unmodified."""
        if not self._running:
            return frame

        now = time.time()
        last = self._last_check.get(participant_id, 0)
        if now - last < INFERENCE_INTERVAL:
            return frame  # Rate limit — age estimation is expensive

        self._last_check[participant_id] = now

        result = await asyncio.get_event_loop().run_in_executor(
            None, self._run_inference, frame
        )

        if result is None:
            return frame

        age = result["age"]
        bucket = age_to_bucket(age)
        is_minor = age < MINOR_AGE_THRESHOLD

        self._last_age[participant_id] = age

        logger.info(f"🎂 Age detected: {participant_id} → ~{age} ({bucket})")

        age_event = AgeDetectedEvent(
            participant_id=participant_id,
            estimated_age=age,
            age_bucket=bucket,
            is_minor=is_minor,
        )

        await self._emit(age_event, emit_event)

        # Fire minor event only once per participant per session
        if is_minor and participant_id not in self._minor_alerted:
            self._minor_alerted.add(participant_id)
            logger.warning(f"🚨 MINOR detected: {participant_id} | estimated age ~{age}")
            minor_event = MinorDetectedEvent(
                participant_id=participant_id,
                estimated_age=age,
                age_bucket=bucket,
            )
            await self._emit(minor_event, emit_event)

        return frame

    async def _emit(self, event, emit_event):
        """Fire event via Vision Agents emit_event and registered callbacks."""
        if emit_event:
            try:
                await emit_event(event)
            except Exception as e:
                logger.error(f"emit_event error: {e}")
        for cb in self._event_callbacks:
            try:
                await cb(event)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    # ── Inference backends ───────────────────────────────────────────────────

    def _run_inference(self, frame: np.ndarray) -> Optional[dict]:
        method, model = self._backend
        if method == "deepface":
            return self._deepface_inference(frame, model)
        return self._heuristic_inference(frame)

    def _deepface_inference(self, frame: np.ndarray, DeepFace) -> Optional[dict]:
        """Use DeepFace to estimate age. Returns dict with 'age' key."""
        try:
            results = DeepFace.analyze(
                img_path=frame,
                actions=["age"],
                enforce_detection=False,  # Don't raise if no face found
                silent=True,
            )
            if not results:
                return None
            # results is a list; take the most prominent face
            face = results[0] if isinstance(results, list) else results
            age = int(face.get("age", 0))
            if age == 0:
                return None
            return {"age": age}
        except Exception as e:
            logger.debug(f"DeepFace inference error: {e}")
            return None

    def _heuristic_inference(self, frame: np.ndarray) -> Optional[dict]:
        """
        Fallback heuristic using OpenCV face detection + skin smoothness proxy.
        Younger faces tend to have smoother, more uniform skin texture.
        This is very rough — use DeepFace for production.
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            if len(faces) == 0:
                return None

            # Take the largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_roi = gray[y:y+h, x:x+w]

            # Skin texture proxy: Laplacian variance
            # Higher variance → more texture → older
            lap_var = cv2.Laplacian(face_roi, cv2.CV_64F).var()

            # Very rough mapping (calibrated empirically, highly imperfect)
            if lap_var < 80:
                age = 10   # Very smooth → child
            elif lap_var < 200:
                age = 16   # Smooth → teen
            elif lap_var < 400:
                age = 25   # Moderate → young adult
            elif lap_var < 700:
                age = 38   # More texture → adult
            else:
                age = 55   # High texture → older adult

            logger.debug(f"Heuristic age estimate: lap_var={lap_var:.1f} → age~{age}")
            return {"age": age}
        except Exception as e:
            logger.debug(f"Heuristic inference error: {e}")
            return None

    # ── Public API ───────────────────────────────────────────────────────────

    def get_estimated_age(self, participant_id: str) -> Optional[int]:
        """Get the last known estimated age for a participant."""
        return self._last_age.get(participant_id)

    def is_minor(self, participant_id: str) -> bool:
        """Returns True if the last detected age was under the minor threshold."""
        age = self._last_age.get(participant_id)
        return age is not None and age < MINOR_AGE_THRESHOLD

    def register_event_callback(self, callback):
        self._event_callbacks.append(callback)

    def clear_minor_alert(self, participant_id: str):
        """Call this if a participant leaves, so re-joining re-triggers detection."""
        self._minor_alerted.discard(participant_id)
        self._last_age.pop(participant_id, None)
        self._last_check.pop(participant_id, None)

    async def stop_processing(self):
        self._running = False

    async def close(self):
        self._running = False
        logger.info("AgeDetectorProcessor closed.")