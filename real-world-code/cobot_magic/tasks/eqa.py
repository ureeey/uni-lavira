"""
Embodied Question Answering (EQA) task for Cobot Magic.

EQA = navigate to the destination (inherited ``VLNTask.run``) and then answer
the user's question from the current views. The QA phase captures the live
front / left / right RGB frames and asks the LA model for the answer.
"""
from __future__ import annotations

import time

from tasks import register_task
from utils import (
    image_to_base64,
    print_action,
    print_error,
    print_info,
    print_success,
    safe_json_loads,
)
from .vln import VLNTask


@register_task("eqa")
class EQATask(VLNTask):
    """Navigate to the destination, then answer the question from current views."""

    def run(self) -> None:
        super().run()
        # After navigation finishes (or stops), perform the QA step.
        self._answer_question()

    def _answer_question(self) -> None:
        print_info("=== Starting EQA Question Answering Phase ===")

        # Give the cameras a moment to settle on the final view.
        time.sleep(1.0)

        content = [{
            "type": "text",
            "text": (
                "You are a robot agent. You have navigated to the destination "
                "based on the instruction.\n"
                "Now, answer the user's question based on what you see RIGHT NOW.\n\n"
                f'User Question: "{self.instruction}"\n\n'
                "Below are the views from your current location.\n"
                "Analyze them carefully to answer the question."
            ),
        }]

        views = [
            ("Front View", "current_front_rgb"),
            ("Left View", "current_left_rgb"),
            ("Right View", "current_right_rgb"),
        ]
        with self.robot.image_lock:
            frames = [(label, getattr(self.robot, attr, None)) for label, attr in views]

        for label, rgb in frames:
            if rgb is not None:
                b64 = image_to_base64(rgb)
                content.append({"type": "text", "text": label})
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })

        content.append({
            "type": "text",
            "text": (
                "Response format (JSON):\n"
                '{\n'
                '    "reasoning": "I see...",\n'
                '    "answer": "Your direct answer"\n'
                '}'
            ),
        })

        print_action("Thinking about the answer...")
        try:
            output_text, _ = self.client.generate_with_la(
                [{"role": "user", "content": content}],
                max_new_tokens=1024, temperature=0,
            )
            result = safe_json_loads(output_text.replace("'", '"'))
            final_answer = result.get("answer", output_text)

            print_success(f"QUESTION: {self.instruction}")
            print_success(f"FINAL ANSWER: {final_answer}")
        except Exception as exc:  # noqa: BLE001 - QA is best-effort, never crash
            print_error(f"EQA Reasoning failed: {exc}")
