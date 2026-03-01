"""
🦸 Superhero Face Filter Processor
Overlays Marvel/DC hero masks on detected faces using OpenCV.
"""

import asyncio
import logging
import random
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from vision_agents.core import Agent

logger = logging.getLogger(__name__)

# Hero definitions: name → mask image filename + color theme
HERO_CATALOG = {
    # Marvel
    "iron_man":         {"file": "iron_man.png",        "color": (0, 60, 200),   "label": "Iron Man"},
    "spider_man":       {"file": "spider_man.png",      "color": (0, 0, 200),    "label": "Spider-Man"},
    "thor":             {"file": "thor.png",             "color": (200, 180, 0),  "label": "Thor"},
    "captain_america":  {"file": "captain_america.png", "color": (200, 50, 50),  "label": "Captain America"},
    "black_panther":    {"file": "black_panther.png",   "color": (80, 0, 80),    "label": "Black Panther"},
    "hulk":             {"file": "hulk.png",             "color": (0, 160, 0),    "label": "Hulk"},
    # DC
    "batman":           {"file": "batman.png",           "color": (40, 40, 40),   "label": "Batman"},
    "wonder_woman":     {"file": "wonder_woman.png",     "color": (0, 30, 180),   "label": "Wonder Woman"},
    "superman":         {"file": "superman.png",         "color": (200, 30, 30),  "label": "Superman"},
    "the_flash":        {"file": "the_flash.png",        "color": (0, 50, 220),   "label": "The Flash"},
}

HERO_ALIASES = {
    "iron man": "iron_man",
    "ironman": "iron_man",
    "spiderman": "spider_man",
    "spider man": "spider_man",
    "spider-man": "spider_man",
    "captain america": "captain_america",
    "cap": "captain_america",
    "black panther": "black_panther",
    "panther": "black_panther",
    "batman": "batman",
    "wonder woman": "wonder_woman",
    "wonderwoman": "wonder_woman",
    "superman": "superman",
    "the flash": "the_flash",
    "flash": "the_flash",
    "thor": "thor",
    "hulk": "hulk",
}

MASKS_DIR = Path(__file__).parent / "assets" / "masks"


class SuperheroFilterProcessor:
    """
    Vision Agents video processor that overlays hero masks on faces.
    Falls back to a stylized color overlay + hero name banner if no PNG mask is found.
    """

    def __init__(self):
        self._agent: Optional["Agent"] = None          # set by attach_agent()
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._participant_filters: dict[str, str] = {}
        self._masks: dict[str, Optional[np.ndarray]] = {}
        self._load_masks()
        self._running = True

    # ── Required by Vision Agents ────────────────────────────────────────────

    def attach_agent(self, agent: "Agent") -> None:
        """Called automatically by Vision Agents when the processor is registered."""
        self._agent = agent

    # ── Mask loading ─────────────────────────────────────────────────────────

    def _load_masks(self):
        """Load PNG overlay masks; store None if file missing (fallback mode)."""
        for key, info in HERO_CATALOG.items():
            path = MASKS_DIR / info["file"]
            if path.exists():
                img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                self._masks[key] = img
                logger.info(f"✅ Loaded mask: {info['label']}")
            else:
                self._masks[key] = None
                logger.warning(f"⚠️  Mask not found: {path} — using fallback overlay")

    # ── Public API ───────────────────────────────────────────────────────────

    def set_filter(self, participant_id: str, hero_name: str) -> bool:
        """Switch a participant's filter. Returns True if hero exists."""
        key = self._resolve_hero_key(hero_name)
        if key:
            self._participant_filters[participant_id] = key
            return True
        return False

    def assign_random_filter(self, participant_id: str) -> str:
        """Assign a random hero filter and return the hero label."""
        key = random.choice(list(HERO_CATALOG.keys()))
        self._participant_filters[participant_id] = key
        return HERO_CATALOG[key]["label"]

    def available_heroes(self) -> list[str]:
        return [info["label"] for info in HERO_CATALOG.values()]

    def _resolve_hero_key(self, name: str) -> Optional[str]:
        n = name.lower().strip()
        if n in HERO_CATALOG:
            return n
        return HERO_ALIASES.get(n)

    # ── Vision Agents processor interface ────────────────────────────────────

    async def process_frame(self, frame: np.ndarray, participant_id: str) -> np.ndarray:
        """Called by Vision Agents for each video frame."""
        if not self._running:
            return frame

        hero_key = self._participant_filters.get(participant_id)
        if not hero_key:
            return frame

        return await asyncio.get_event_loop().run_in_executor(
            None, self._apply_filter_sync, frame, hero_key
        )

    def _apply_filter_sync(self, frame: np.ndarray, hero_key: str) -> np.ndarray:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        result = frame.copy()
        hero   = HERO_CATALOG[hero_key]

        for (x, y, w, h) in faces:
            mask_img = self._masks.get(hero_key)
            if mask_img is not None:
                result = self._overlay_mask(result, mask_img, x, y, w, h)
            else:
                result = self._fallback_overlay(result, x, y, w, h, hero)

        return result

    def _overlay_mask(self, frame: np.ndarray, mask: np.ndarray, x, y, w, h) -> np.ndarray:
        """Resize and alpha-blend the hero mask onto the detected face region."""
        try:
            resized = cv2.resize(mask, (w, h))
            if resized.shape[2] == 4:
                alpha = resized[:, :, 3] / 255.0
                for c in range(3):
                    frame[y:y+h, x:x+w, c] = (
                        alpha * resized[:, :, c] +
                        (1 - alpha) * frame[y:y+h, x:x+w, c]
                    ).astype(np.uint8)
            else:
                frame[y:y+h, x:x+w] = resized
        except Exception as e:
            logger.debug(f"Mask overlay error: {e}")
        return frame

    def _fallback_overlay(self, frame: np.ndarray, x, y, w, h, hero: dict) -> np.ndarray:
        """Stylized color tint + hero name banner as fallback."""
        overlay = frame.copy()
        color   = hero["color"]
        cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
        frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

        label = hero["label"].upper()
        font  = cv2.FONT_HERSHEY_DUPLEX
        scale = w / 200
        thickness = max(1, int(w / 80))
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        tx = x + (w - tw) // 2
        ty = y - 10 if y - 10 > th else y + h + th + 10

        cv2.rectangle(frame, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4), (0, 0, 0), -1)
        cv2.putText(frame, label, (tx, ty), font, scale, color, thickness, cv2.LINE_AA)

        return frame

    async def stop_processing(self) -> None:
        self._running = False

    async def close(self) -> None:
        self._running = False
        logger.info("SuperheroFilterProcessor closed.")