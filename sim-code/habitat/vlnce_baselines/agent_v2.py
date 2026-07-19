"""V2 reasoning agent for rollout_v2.

Inherits infrastructure (visualizer, model, API client) from VLMReasoningAgent
and implements a multi-turn conversation chain for per-step VLM decisions.
Each decision step builds on the previous conversation — the VLM sees the full
dialogue history for that step, avoiding redundant image re-uploads.
"""
import json
import os as _os
import re

import numpy as np
from PIL import Image

from .agent import VLMReasoningAgent
from .utils.api import log_response
from .prompts.prompts_objnav_v2 import (
    PROMPT_IS_TARGET_NEAR,
    PROMPT_IS_TARGET_POSSIBLE,
    PROMPT_IS_TARGET_VISIBLE,
    PROMPT_IS_REPEAT_CURRENT,
    PROMPT_IS_REPEAT_HISTORY,
    PROMPT_POSSIBLE_BBOX,
    PROMPT_TARGET_BBOX,
)

# Request logging: 0=off, 1=incremental (new msgs only), 2=full
from .utils.logging import LOG_REQ, log_req, log_req_full


class VLMReasoningAgentV2(VLMReasoningAgent):
    """Per-step VLM decision agent for rollout_v2.

    Replaces the panorama→LA→VA→planner pipeline with six lightweight methods
    that form a single multi-turn conversation chain. Each step resets the
    conversation; within a step, calls append new user prompts to the existing
    messages so the VLM sees the full context without re-uploading images.
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
        import re
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

    def _start_conversation(self, rgb_image, text):
        """Begin a new conversation with an image and a text prompt."""
        img = self._to_pil(rgb_image)
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.img_to_base64(img)}"}},
            {"type": "text", "text": text},
        ]
        self._messages = [{"role": "user", "content": content}]
        self._req_last_count = 0  # reset incremental log tracker for new conversation

    def _append_user_text(self, text):
        """Append a user text message (no image) to the conversation."""
        self._messages.append({"role": "user", "content": [{"type": "text", "text": text}]})

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
            logger.info(f"--REQ --- [{start}..{len(self._messages)-1}] ---")
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

    def _call_api_and_parse_yes_no(self):
        """Call the VA API and parse a yes/no answer.  Retries up to 3 times."""
        for _attempt in range(3):
            self._log_messages()
            output = self.model.generate(
                messages=self._messages,
                max_new_tokens=256,
                temperature=0.0,
                label="V2",
                extra_body={"enable_thinking": False},
            )
            self._messages.append({"role": "assistant", "content": output})
            self._last_output = output
            log_response("--RESP")
            log_response(output)
            lower = output.strip().lower()
            if lower.startswith("yes") or "yes" in lower[:10]:
                return True
            if lower.startswith("no") or "no" in lower[:10]:
                return False
        return False

    def _call_api_and_parse_bbox(self):
        """Call the VA API and parse a bbox_2d from JSON output.  Retries up to 5 times."""
        w = self.visualizer.width
        h = self.visualizer.height
        for _attempt in range(5):
            self._log_messages()
            output = self.model.generate(
                messages=self._messages,
                max_new_tokens=1024,
                temperature=0.0,
                label="V2",
                extra_body={"enable_thinking": False},
            )
            self._messages.append({"role": "assistant", "content": output})
            self._last_output = output
            log_response("--RESP")
            log_response(output)
            m = re.search(r"\{.*\}", output, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                    bbox_2d = data.get("bbox_2d", [w // 4, h // 4, 3 * w // 4, 3 * h // 4])
                    if len(bbox_2d) >= 4:
                        x1, y1, x2, y2 = [int(v) for v in bbox_2d[:4]]
                        # coords are in per-mille (0-1000)
                        x1 = max(0, min(int(x1 / 1000 * w), w - 1))
                        y1 = max(0, min(int(y1 / 1000 * h), h - 1))
                        x2 = max(x1 + 1, min(int(x2 / 1000 * w), w))
                        y2 = max(y1 + 1, min(int(y2 / 1000 * h), h))
                        return {
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "x": x1, "y": y1,
                            "width": x2 - x1, "height": y2 - y1,
                            "target": "target",
                        }
                except Exception:
                    pass
        # Fallback: center of image
        return {
            "x1": w // 4, "y1": h // 4, "x2": 3 * w // 4, "y2": 3 * h // 4,
            "x": w // 4, "y": h // 4,
            "width": w // 2, "height": h // 2,
            "target": "fallback",
        }

    # ------------------------------------------------------------------
    # Public decision interface (called by rollout_v2)
    # ------------------------------------------------------------------

    def is_target_visible(self, target: str, rgb_image: np.ndarray) -> bool:
        """Check if *target* is visible in the current RGB frame.

        Starts a new conversation: sends the image + yes/no question to the VLM.
        """
        name = self._target_name(target)
        self._start_conversation(rgb_image, PROMPT_IS_TARGET_VISIBLE.format(target=name))
        return self._call_api_and_parse_yes_no()

    def is_target_near(self, target: str, rgb_image: np.ndarray) -> bool:
        """Check if the visible target is close enough to stop.

        Appends to the conversation started by is_target_visible.
        """
        name = self._target_name(target)
        self._append_user_text(PROMPT_IS_TARGET_NEAR.format(target=name))
        return self._call_api_and_parse_yes_no()

    def target_bbox(self, target: str, rgb_image: np.ndarray) -> dict:
        """Return a bbox for the clearly visible target.

        Appends to the conversation chain: is_target_visible → is_target_near.
        """
        name = self._target_name(target)
        self._append_user_text(PROMPT_TARGET_BBOX.format(target=name))
        return self._call_api_and_parse_bbox()

    def is_target_possible(self, target: str, rgb_image: np.ndarray) -> bool:
        """Check if the frame contains passable, explorable area that might lead
        to the target.

        Appends to the conversation started by is_target_visible (which returned
        False, so the VLM already knows the target is not visible).
        """
        name = self._target_name(target)
        self._append_user_text(PROMPT_IS_TARGET_POSSIBLE.format(target=name))
        return self._call_api_and_parse_yes_no()

    def possible_bbox(self, target: str, rgb_image: np.ndarray) -> dict:
        """Return a bbox for the most promising exploration area.

        Appends to the conversation chain: is_target_visible → is_target_possible.
        """
        name = self._target_name(target)
        self._append_user_text(PROMPT_POSSIBLE_BBOX.format(target=name))
        return self._call_api_and_parse_bbox()

    def is_repeat(self, history_images: list, current_image: np.ndarray) -> bool:
        """Check if the current bbox-annotated frame is a repeat of previously
        seen frames (loop detection).

        Appends two message blocks to the conversation:
        1. Current annotated photo with label
        2. History annotated photos with comparison prompt
        """
        # Block 1: current photo with marked area
        img = self._to_pil(current_image)
        self._messages.append({"role": "user", "content": [
            {"type": "text", "text": PROMPT_IS_REPEAT_CURRENT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.img_to_base64(img)}"}},
        ]})

        # Block 2: history marked photos + comparison prompt
        hist_blocks = [{"type": "text", "text": PROMPT_IS_REPEAT_HISTORY}]
        if history_images:
            for hist_img in history_images:
                h_img = self._to_pil(hist_img)
                hist_blocks.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.img_to_base64(h_img)}"}}
                )
        self._messages.append({"role": "user", "content": hist_blocks})

        return self._call_api_and_parse_yes_no()
