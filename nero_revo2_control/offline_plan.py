"""Nero home-to-grasp kinematics study; never connects to hardware.

Only synthetic flange poses are solved while the real hand-eye and grasp-TCP
transforms remain unaccepted. Joint interpolation is checked separately from
scene collision, so even a successful IK result is never marked executable.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from nero_revo2_control.kinematics import load_model


READY_HOME_DEG = (55.0, -78.0, 80.0, -45.0, 130.0, -30.0, 40.0)
READY_HOME_RAD = tuple(math.radians(value) for value in READY_HOME_DEG)

UNVERIFIED_BLOCKERS = (
    "Accepted, physically checked T_base_camera is unavailable",
    "Measured flange-to-right-Revo2 grasp TCP is unavailable",
    "Nero, Revo2, fixtures, table, and cable collision model is incomplete",
)


def _degrees(joints_rad):
    return [round(math.degrees(value), 6) for value in joints_rad]


def _matrix_list(matrix):
    return [[round(float(value), 9) for value in row] for row in matrix]


def _result_field(result, name):
    """Fail clearly if the kinematics contract changes."""
    return getattr(result, name)


def _stage(model, name, target_matrix, seed_rad):
    solution = model.ik(target_matrix, seed_rad)
    joints = _result_field(solution, "joints_rad")
    stage = {
        "name": name,
        "target_frame": "base_link_to_link7",
        "target_matrix": _matrix_list(target_matrix),
        "ik": {
            "success": bool(_result_field(solution, "success")),
            "joints_deg": _degrees(joints),
            "position_error_m": float(_result_field(solution, "position_error_m")),
            "orientation_error_rad": float(
                _result_field(solution, "orientation_error_rad")
            ),
            "iterations": int(_result_field(solution, "iterations")),
            "reason": str(_result_field(solution, "reason")),
        },
        "joint_path": None,
    }
    if stage["ik"]["success"]:
        path = model.check_joint_path(
            seed_rad, joints, max_sample_step_rad=math.radians(1.0)
        )
        stage["joint_path"] = {
            "type": "joint_linear_interpolation",
            "max_sample_step_deg": 1.0,
            "sample_count": int(_result_field(path, "sample_count")),
            "sampled_joints_deg": [
                _degrees(sample) for sample in _result_field(path, "samples_rad")
            ],
            "joint_limits_ok": bool(_result_field(path, "joint_limits_ok")),
            "continuity_ok": bool(_result_field(path, "continuity_ok")),
            "singularity_warning": bool(_result_field(path, "singularity_warning")),
            "scene_collision_verified": False,
            "self_collision_verified": False,
            "reason": str(_result_field(path, "reason")),
        }
    return stage, joints


def plan_synthetic():
    """Generate two nearby 6D poses via URDF FK, then solve them offline."""
    model = load_model()
    pregrasp = list(READY_HOME_RAD)
    pregrasp[0] += math.radians(2.0)
    grasp = list(pregrasp)
    grasp[6] += math.radians(2.0)
    seed = READY_HOME_RAD
    stages = []
    for name, target_joints in (("pregrasp", pregrasp), ("grasp", grasp)):
        target = model.fk(target_joints)
        stage, seed = _stage(model, name, target, seed)
        stages.append(stage)
        if not stage["ik"]["success"]:
            break
    kinematics_ok = len(stages) == 2 and all(
        stage["ik"]["success"]
        and stage["joint_path"] is not None
        and stage["joint_path"]["joint_limits_ok"]
        and stage["joint_path"]["continuity_ok"]
        for stage in stages
    )
    return {
        "schema_version": 1,
        "status": "synthetic_kinematics_complete"
        if kinematics_ok
        else "kinematics_failed",
        "source": "synthetic_fk_only",
        "model": "Nero seven-axis URDF",
        "home_joints_deg": list(READY_HOME_DEG),
        "sequence": ["home", "pregrasp", "grasp"],
        "stages": stages,
        "kinematics_ok": kinematics_ok,
        "scene_collision_verified": False,
        "self_collision_verified": False,
        "real_hand_eye_verified": False,
        "grasp_tcp_verified": False,
        "executable": False,
        "motion_sent": False,
        "blockers": list(UNVERIFIED_BLOCKERS),
    }


def plan_from_localization(path):
    """Inspect a localization plan; never promote an unaccepted target."""
    source_path = Path(path)
    data = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("localization plan must be a JSON object")
    source_blockers = data.get("blockers")
    if source_blockers is None:
        source_blockers = []
    if not isinstance(source_blockers, list) or not all(
        isinstance(value, str) for value in source_blockers
    ):
        raise ValueError("localization blockers must be a list of strings")
    blockers = list(UNVERIFIED_BLOCKERS)
    if data.get("base_targets") is None:
        blockers.insert(0, "Localization has no base-frame pregrasp/grasp targets")
    return {
        "schema_version": 1,
        "status": "blocked",
        "source": "localization_plan_unaccepted",
        "input_plan": str(source_path),
        "sequence": ["home", "pregrasp", "grasp"],
        "base_targets_available": data.get("base_targets") is not None,
        "input_blockers": source_blockers,
        "stages": [],
        "kinematics_ok": False,
        "scene_collision_verified": False,
        "self_collision_verified": False,
        "real_hand_eye_verified": False,
        "grasp_tcp_verified": False,
        "executable": False,
        "motion_sent": False,
        "blockers": blockers,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("synthetic", help="offline FK-to-IK two-stage study")
    localization = commands.add_parser(
        "from-localization", help="inspect cup localization plan without motion"
    )
    localization.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "synthetic":
        result = plan_synthetic()
    else:
        result = plan_from_localization(args.plan)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] != "kinematics_failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
