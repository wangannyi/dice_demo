"""Held-cup authorization is distinct from the empty-hand joint-test contract."""

import hashlib
import json
from pathlib import Path


def validate_held_request(request):
    path = Path(request["grasp_receipt_path"])
    if (
        request["input_hashes"].get(str(path))
        != hashlib.sha256(path.read_bytes()).hexdigest()
    ):
        raise ValueError("Grasp receipt missing from reviewed inputs")
    receipt = json.loads(path.read_text())
    command = receipt.get("hand_command", {})
    if (
        not receipt.get("success")
        or command.get("target_0_100") != request["grip_targets_0_100"]
    ):
        raise ValueError("No successful configured grasp command")
    if (command.get("position_target_reached") is False
            and not (command.get("position_required") is False
                     and command.get("completion_basis") == "command_duration_only_position_not_required")):
        raise ValueError("Grasp feedback rejected target")
    if (
        request["table_screen"].get("held_cup_min_mm", -1)
        < request["held_cup_margin_mm"]
    ):
        raise ValueError("Held cup path lacks table clearance")
