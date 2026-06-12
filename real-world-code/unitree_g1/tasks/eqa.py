"""Embodied Question Answering (EQA) task.

EQA = navigate to the destination (inherited ``VLNTask.run``) and then answer
the user's question from the current view.  Ported verbatim from the G1 source:
the QA phase refreshes the single front camera via ``update_camera_data`` and
asks the LA model for the answer using ``self.robot.rgb_image``.  The inline
secondary-client OpenAI completion call is replaced by
``self.client.generate_with_la`` with the SAME per-call params
(LA ``temperature=0, max_new_tokens=1024``).
"""
import time

from colorama import Fore

from config import Config
from tasks import register_task
from utils import (
    numpy_to_base64,
    print_action,
    print_error,
    print_info,
    safe_json_loads,
    save_output,
)
from .vln import VLNTask


@register_task("eqa")
class EQATask(VLNTask):
    """Embodied Question Answering Task."""

    def run(self):
        super().run()
        # After navigation finishes (or stops), perform QA.
        self._answer_question()

    def _answer_question(self):
        """After navigation, answer the question based on the current view."""
        print_info("=== Starting EQA Question Answering Phase ===")

        time.sleep(1.0)

        # Capture a fresh view from the current position (single front camera).
        self.robot.update_camera_data()

        content = [
            {
                "type": "text",
                "text": f"""
            You are a humanoid robot agent. You have navigated to the destination based on the instruction.
            Now, answer the user's question based on what you see RIGHT NOW.

            User Question: "{self.instruction}"

            Below is the view from your current location.
            Analyze it carefully to answer the question.
            """,
            }
        ]

        # Use the current camera view (single camera on G1).
        rgb = self.robot.rgb_image
        if rgb is not None:
            b64 = numpy_to_base64(rgb)
            content.append({"type": "text", "text": "Current View:"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )

        content.append(
            {
                "type": "text",
                "text": """
            Response format (JSON):
            {
                "reasoning": "I see...",
                "answer": "Your direct answer"
            }
            """,
            }
        )

        print_action("Thinking about the answer...")
        try:
            output_text, _ = self.client.generate_with_la(
                [{"role": "user", "content": content}],
                max_new_tokens=1024,
                temperature=0,
            )
            result = safe_json_loads(output_text.replace("'", '"'))
            final_answer = result.get("answer", output_text)

            print(Fore.MAGENTA + "=" * 40)
            print(Fore.MAGENTA + f"QUESTION: {self.instruction}")
            print(Fore.MAGENTA + f"FINAL ANSWER: {final_answer}")
            print(Fore.MAGENTA + "=" * 40)

            save_output(
                Config.LOG_DIR,
                "final_eqa_result.json",
                {"question": self.instruction, "answer": final_answer},
            )

        except Exception as e:
            print_error(f"EQA Reasoning failed: {e}")
