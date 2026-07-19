"""V3 reasoning agent for rollout_v3.

Inherits infrastructure (visualizer, model, API client) from VLMReasoningAgent
and implements single-call VLM decisions.  Unlike v2's multi-turn conversation
chain, v3 makes one standalone ``judge()`` call with 4-directional views to
simultaneously classify the situation and return bbox candidates.  An optional
``select_one()`` call picks the best exploration area when multiple candidates
exist.
"""
import json
import os as _os
import re

import numpy as np
from PIL import Image

from .agent import VLMReasoningAgent
from .utils.api import log_response
from .prompts.prompts_objnav_v3 import (
    PROMPT_JUDGE,
    PROMPT_SELECT_ONE,
    PROMPT_SELECT_ONE_CANDIDATES_HEADER,
    PROMPT_SELECT_ONE_HISTORY_HEADER,
)

# Request logging: 0=off, 1=incremental (new msgs only), 2=full
from .utils.logging import LOG_REQ, log_req, log_req_full


class VLMReasoningAgentV3(VLMReasoningAgent):
    """Single-call VLM decision agent for rollout_v3.

    Uses a single multimodal request with 4-directional views to make
    STOP / APPROACH / EXPLORE / OTHER decisions, replacing v2's multi-turn
    yes/no+bbox conversation chain.
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

    def _start_conversation(self, images, text):
        """Begin a new conversation with multiple images and a text prompt.

        Parameters
        ----------
        images : list of np.ndarray
            RGB images to include in the message.
        text : str
            Prompt text (placed after the images).
        """
        content = []
        for img in images:
            pil_img = self._to_pil(img)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{self.img_to_base64(pil_img)}"
                },
            })
        content.append({"type": "text", "text": text})
        self._messages = [{"role": "user", "content": content}]
        self._req_last_count = 0  # reset incremental log tracker

    def _append_images_and_text(self, images, text):
        """Append a user message with images and text to the conversation."""
        content = []
        for img in images:
            pil_img = self._to_pil(img)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{self.img_to_base64(pil_img)}"
                },
            })
        content.append({"type": "text", "text": text})
        self._messages.append({"role": "user", "content": content})

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

    def _call_api_and_parse_judge_json(self):
        """Call the VA API and parse the judge response (letter + JSON).

        Retries up to 5 times.  Returns ``(plan, regions)`` where *plan* is
        one of ``"STOP"`` / ``"APPROACH"`` / ``"EXPLORE"`` / ``"OTHER"``
        and *regions* is the hierarchical list.
        """
        w = self.visualizer.width
        h = self.visualizer.height

        for _attempt in range(5):
            self._log_messages()
            output = self.model.generate(
                messages=self._messages,
                max_new_tokens=2048,
                temperature=0.0,
                label="V3",
                extra_body={"enable_thinking": False},
            )
            self._messages.append({"role": "assistant", "content": output})
            self._last_output = output
            log_response("--RESP")
            log_response(output)

            # Extract JSON block
            m = re.search(r"\{.*\}", output, re.DOTALL)
            if not m:
                continue
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                continue

            plan_letter = data.get("plan", "").strip().upper()
            if plan_letter not in ("A", "B", "C", "D"):
                continue  # invalid plan letter, retry
            regions_raw = data.get("regions", [])

            plan_map = {"A": "STOP", "B": "APPROACH", "C": "EXPLORE", "D": "OTHER"}
            plan = plan_map[plan_letter]

            # Validate and convert regions
            regions = []
            parse_ok = True
            for item in regions_raw:
                frame_idx = item.get("frame_idx")
                if frame_idx is None or not isinstance(frame_idx, int):
                    parse_ok = False
                    break
                frame_regions = []
                for reg in item.get("regions", []):
                    bbox_2d = reg.get("bbox")
                    if bbox_2d is None or len(bbox_2d) < 4:
                        parse_ok = False
                        break
                    idx = reg.get("idx")
                    if idx is None or not isinstance(idx, int):
                        parse_ok = False
                        break
                    x1 = max(0, min(int(bbox_2d[0] / 1000 * w), w - 1))
                    y1 = max(0, min(int(bbox_2d[1] / 1000 * h), h - 1))
                    x2 = max(x1 + 1, min(int(bbox_2d[2] / 1000 * w), w))
                    y2 = max(y1 + 1, min(int(bbox_2d[3] / 1000 * h), h))
                    frame_regions.append({
                        "idx": idx,
                        "bbox": [x1, y1, x2, y2],
                    })
                if frame_regions:
                    regions.append({"frame_idx": frame_idx, "regions": frame_regions})

            if not parse_ok:
                continue

            # Plan-specific constraints: A/B must have exactly 1 frame+region
            total_regions = sum(len(item.get("regions", [])) for item in regions)
            if plan in ("STOP", "APPROACH"):
                if len(regions) != 1 or total_regions != 1:
                    continue
            elif plan == "EXPLORE":
                if len(regions) == 0 or total_regions == 0:
                    continue

            return plan, regions

        # Fallback: OTHER with empty regions
        return "OTHER", []

    def _call_api_and_parse_select_json(self, hier_list=None):
        """Call the VA API and parse the select_one response.

        Retries up to 3 times.  If *hier_list* is provided, validates
        that the returned ``(frame_idx, bbox_idx)`` actually exists in
        the candidate list and retries on mismatch.

        Returns ``(frame_idx, bbox_idx)`` or ``(None, None)``.
        """
        for _attempt in range(3):
            self._log_messages()
            output = self.model.generate(
                messages=self._messages,
                max_new_tokens=512,
                temperature=0.0,
                label="V3",
                extra_body={"enable_thinking": False},
            )
            self._messages.append({"role": "assistant", "content": output})
            self._last_output = output
            log_response("--RESP")
            log_response(output)

            m = re.search(r"\{.*\}", output, re.DOTALL)
            if not m:
                continue
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                continue

            frame_idx = data.get("frame_idx")
            bbox_idx = data.get("bbox_idx")
            if frame_idx is None and bbox_idx is None:
                return None, None
            if frame_idx is not None and bbox_idx is not None:
                frame_idx = int(frame_idx)
                bbox_idx = int(bbox_idx)
                # Validate against hier_list if provided
                if hier_list is not None:
                    valid = any(
                        item['frame_idx'] == frame_idx and
                        any(reg.get('idx') == bbox_idx for reg in item.get('regions', []))
                        for item in hier_list
                    )
                    if not valid:
                        from habitat import logger
                        logger.info(f"--V3 select_one retry: invalid ({frame_idx},{bbox_idx}) not in candidates")
                        continue
                return frame_idx, bbox_idx

        return None, None

    # ------------------------------------------------------------------
    # Public decision interface (called by rollout_v3)
    # ------------------------------------------------------------------

    def judge(self, target: str, four_rgb_images: list,
              came_from_direction: int = None):
        """Classify the situation from 4-directional views and return bbox candidates.

        Parameters
        ----------
        target : str
            ObjectNav instruction (e.g. ``"Find the pillow"``).
        four_rgb_images : list of np.ndarray
            RGB images in order: [front, right, back, left].
        came_from_direction : int or None
            Which frame (0-3) points toward the previous position.
            ``None`` on the first call.

        Returns
        -------
        plan : str
            One of ``"STOP"`` / ``"APPROACH"`` / ``"EXPLORE"`` / ``"OTHER"``.
        regions : list
            Hierarchical list of ``[{frame_idx, regions: [{idx, bbox}, ...]}, ...]``.
        """
        name = self._target_name(target)

        # Build came-from hint
        direction_names = {0: "Front", 1: "Right", 2: "Back", 3: "Left"}
        if came_from_direction is not None:
            dir_name = direction_names.get(came_from_direction, "Unknown")
            came_from_hint = (
                f"\n(Note: the {dir_name} view shows the direction "
                f"toward the previous position — avoid going this way.)"
            )
        else:
            came_from_hint = ""

        prompt = PROMPT_JUDGE.format(target=name, came_from_hint=came_from_hint)
        self._start_conversation(four_rgb_images, prompt)
        return self._call_api_and_parse_judge_json()

    def select_one(self, target: str, bbox_history_images: list,
                   f_ann_list: list, hierarchical_list: list):
        """Pick the best unexplored candidate from the exploration areas.

        Parameters
        ----------
        target : str
            ObjectNav instruction.
        bbox_history_images : list of np.ndarray
            Previously explored bbox-annotated images.
        f_ann_list : list of np.ndarray
            Newly annotated candidate images (one per region).
        hierarchical_list : list
            The hierarchical list from ``judge()``, used for reference in the prompt.

        Returns
        -------
        frame_idx : int or None
            Index of the selected frame, or ``None`` if all explored.
        bbox_idx : int or None
            Index of the selected region, or ``None`` if all explored.
        """
        name = self._target_name(target)

        # Build prompt text
        hier_json = json.dumps(hierarchical_list)
        text_parts = [
            PROMPT_SELECT_ONE.format(target=name),
            "",
            PROMPT_SELECT_ONE_CANDIDATES_HEADER,
            "```json",
            hier_json,
            "```",
        ]

        prompt_text = "\n".join(text_parts)

        # Build the message: start with candidate images
        self._start_conversation(f_ann_list, prompt_text)

        # If there are history images, append them
        if bbox_history_images:
            self._append_images_and_text(
                bbox_history_images,
                PROMPT_SELECT_ONE_HISTORY_HEADER
            )

        return self._call_api_and_parse_select_json(hier_list=hierarchical_list)
