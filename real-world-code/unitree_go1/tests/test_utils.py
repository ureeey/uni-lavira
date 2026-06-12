import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import utils


def test_safe_json_loads_plain():
    assert utils.safe_json_loads('{"a": 1}') == {"a": 1}


def test_safe_json_loads_fenced():
    assert utils.safe_json_loads('```json\n{"a": 1}\n```') == {"a": 1}


def test_safe_json_loads_trailing_text():
    assert utils.safe_json_loads('reasoning... {"action": "front"} done')["action"] == "front"
