"""V4 reasoning agent for rollout_v4.

Inherits infrastructure (visualizer, model, API client) from VLMReasoningAgent
and implements a single ``decide()`` call that merges v3's judge + select_one
into one step.  The model sees 4-directional views + optional bbox history and
returns a single-line decision with at most one bbox.
"""
import re

import numpy as np
from PIL import Image

from .agent import VLMReasoningAgent
from .utils.api import log_response
from .prompts.prompts_objnav_v4 import (
    PROMPT_V4,
    PROMPT_V4_HISTORY_HEADER,
)

# Request logging
from .utils.logging import LOG_REQ, log_req, log_req_full


class VLMReasoningAgentV4(VLMReasoningAgent):
    """Single-call VLM decision agent for rollout_v4.

    Merges judge + select_one: one multimodal request with 4-directional
    views and optional bbox history, returning a single plan + bbox.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _target_name(instruction: str) -> str:
        """Extract the target object name from an ObjectNav instruction.

        ``"Find the pillow"`` → ``"pillow"``, ``"Find chair"`` → ``"chair"``.
        Falls back to the full instruction if the pattern doesn't match.
        """
        m = re.search(r"^Find\s+(?:the\s+)?(.+)", instruction, re.IGNORECASE)
        return m.group(1).strip() if m else instruction

    @staticmethod
    def _to_pil(rgb_image):
        """Convert a numpy RGB array or PIL Image to a PIL Image."""
        if isinstance(rgb_image, np.ndarray):
            if rgb_image.dtype != np.uint8:
                rgb_image = (rgb_image * 255).astype(np.uint8)
            return Image.fromarray(rgb_image)
        return rgb_image

    def _log_messages(self):
        """Print the conversation content for debugging.

        Controlled by LAVIRA_LOG_REQ:
          0 = off,  1 = incremental (new messages only),  2 = full.
        """
        if LOG_REQ == 0:
            self._req_last_count = len(self._messages)
            return
        from habitat import logger
        total_kb = 0
        last = getattr(self, '_req_last_count', 0)
        start = 0 if LOG_REQ >= 2 else last
        if start < len(self._messages):
            logger.info(f"--REQ [{start}..{len(self._messages) - 1}] ---")
        for i in range(start, len(self._messages)):
            msg = self._messages[i]
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, str):
                logger.info(f"--REQ [{i}] {role}: {content}")
                total_kb += len(content) / 1024
            elif isinstance(content, list):
                imgs_kb = 0
                for item in content:
                    if item.get("type") == "text":
                        t = item["text"]
                        logger.info(f"--REQ [{i}] {role} text: {t}")
                        total_kb += len(t) / 1024
                    elif item.get("type") == "image_url":
                        url = item["image_url"]["url"]
                        size_kb = len(url) / 1024
                        imgs_kb += size_kb
                total_kb += imgs_kb
                if imgs_kb > 0:
                    logger.info(f"--REQ [{i}] {role} image: {imgs_kb:.0f} KB")
        self._req_last_count = len(self._messages)
        if total_kb > 0:
            logger.info(f"--REQ --- {total_kb:.0f} KB ---")

    # ------------------------------------------------------------------
    # Public decision interface (called by rollout_v4)
    # ------------------------------------------------------------------

    def decide(self, target: str, four_rgb_images: list,
               came_from_direction: int = None,
               bbox_history_images: list = None):
        """Single-call VLM decision: classify + pick best bbox.

        Parameters
        ----------
        target : str
            ObjectNav instruction (e.g. ``"Find the pillow"``).
        four_rgb_images : list of np.ndarray
            RGB images in order: [front, right, back, left].
        came_from_direction : int or None
            Which frame (0-3) points toward the previous position.
            ``None`` on the first call.
        bbox_history_images : list of np.ndarray or None
            Previously explored bbox-annotated images to show the model.

        Returns
        -------
        plan : str
            One of ``"STOP"`` / ``"APPROACH"`` / ``"EXPLORE"`` / ``"OTHER"``.
        frame_idx : int or None
            Index of the frame (0=front, 1=right, 2=back, 3=left) for the bbox,
            or ``None`` for ``"OTHER"``.
        bbox_px : list of int or None
            ``[x1, y1, x2, y2]`` in pixel coordinates, or ``None`` for ``"OTHER"``.

        Raises
        ------
        RuntimeError
            If the model response does not match the expected format.
        """
        name = self._target_name(target)
        w = self.visualizer.width
        h = self.visualizer.height

        # Build came-from fact (factual statement, no "avoid" constraint)
        direction_names = {0: "front", 1: "right", 2: "back", 3: "left"}
        if came_from_direction is not None:
            dir_name = direction_names.get(came_from_direction, "unknown")
            came_from_fact = f"Came from: {dir_name}.\n"
        else:
            came_from_fact = ""

        prompt = PROMPT_V4.format(target=name, came_from_fact=came_from_fact)

        # Build messages: system + user(4 images) + optional history
        self._messages = [{"role": "system", "content": "Output only the result. No analysis."}]

        content = [{"type": "text", "text": prompt}]
        for img in four_rgb_images:
            pil_img = self._to_pil(img)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{self.img_to_base64(pil_img)}"
                },
            })
        self._messages.append({"role": "user", "content": content})
        self._req_last_count = 0

        # Append history images as second message if available
        if bbox_history_images:
            hist_content = [{"type": "text", "text": PROMPT_V4_HISTORY_HEADER}]
            for img in bbox_history_images:
                pil_img = self._to_pil(img)
                hist_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{self.img_to_base64(pil_img)}"
                    },
                })
            self._messages.append({"role": "user", "content": hist_content})

        # Call model
        self._log_messages()
        output = self.model.generate(
            messages=self._messages,
            max_new_tokens=2048,
            temperature=0.0,
            label="V4",
            extra_body={"enable_thinking": False},
        )
        self._messages.append({"role": "assistant", "content": output})
        log_response("--RESP")
        log_response(output)

        # Strict parse — no retries, no fallback
        plan, frame_idx, bbox_px = self._parse_response(output, w, h)
        return plan, frame_idx, bbox_px

    def _parse_response(self, output: str, w: int, h: int):
        """Strict one-shot parse of the model response.

        Expected format:
          ``<A|B|C>,<front|right|back|left>,<x1>,<y1>,<x2>,<y2>``
          or ``D``

        Raises RuntimeError on any mismatch.
        """
        dir_map = {"front": 0, "right": 1, "back": 2, "left": 3}
        plan_map = {"A": "STOP", "B": "APPROACH", "C": "EXPLORE", "D": "OTHER"}

        # Take the last non-empty line
        lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
        if not lines:
            raise RuntimeError(f"V4 parse error: empty response")

        last_line = lines[-1]

        # Try D (single letter, no coordinates)
        if last_line == "D":
            return "OTHER", None, None

        # Try <letter>,<dir>,<x1>,<y1>,<x2>,<y2>
        m = re.match(
            r"^([A-D]),(front|right|back|left),(\d{1,4}),(\d{1,4}),(\d{1,4}),(\d{1,4})$",
            last_line,
        )
        if not m:
            raise RuntimeError(f"V4 parse error: {output!r}")

        letter = m.group(1)
        direction = m.group(2)
        x1 = int(m.group(3))
        y1 = int(m.group(4))
        x2 = int(m.group(5))
        y2 = int(m.group(6))

        # Validate
        if letter == "D":
            raise RuntimeError(f"V4 parse error: D must not have coordinates: {output!r}")
        if not (0 <= x1 <= 1000 and 0 <= y1 <= 1000 and 0 <= x2 <= 1000 and 0 <= y2 <= 1000):
            raise RuntimeError(f"V4 parse error: coordinates out of 0-1000 range: {output!r}")

        plan = plan_map[letter]
        frame_idx = dir_map[direction]

        # Convert per-mille to pixel coordinates
        px1 = max(0, min(int(x1 / 1000 * w), w - 1))
        py1 = max(0, min(int(y1 / 1000 * h), h - 1))
        px2 = max(px1 + 1, min(int(x2 / 1000 * w), w))
        py2 = max(py1 + 1, min(int(y2 / 1000 * h), h))

        return plan, frame_idx, [px1, py1, px2, py2]
