"""Gesture registry: scan configs/actions/gestures/*.json and merge flat.

Each group file keeps the result_feedback.json shape (top-level defaults +
gestures + aliases); defaults only apply to gestures in the same file. Groups
load in filename order; a duplicate name or an invalid recipe rejects the
whole later file, never an earlier one, and never the other groups.
"""
import json
from pathlib import Path

from scripts.result_feedback import recipe_for

ROOT = Path(__file__).resolve().parents[1]
GESTURES_DIR = ROOT / "configs/actions/gestures"


class ActionRegistry:
    def __init__(self, groups, errors):
        # groups: list of (path, config), accepted in filename order.
        self.groups = groups
        self.errors = errors

    def names(self):
        result = []
        for _, config in self.groups:
            result += list(config.get("gestures", {}))
            result += list(config.get("aliases", {}))
        return result

    def recipe(self, name):
        for _, config in self.groups:
            if name in config.get("gestures", {}) or name in config.get("aliases", {}):
                return recipe_for(config, name)
        raise ValueError("未知动作：" + str(name) + "；可选：" + ", ".join(sorted(self.names())))


def _group_names(config):
    return set(config.get("gestures", {})) | set(config.get("aliases", {}))


def load_registry(source=GESTURES_DIR):
    """Load a directory of gesture groups (or one explicit .json file)."""
    source = Path(source)
    if source.is_file():
        files = [source]
    elif source.is_dir():
        files = sorted(source.glob("*.json"))
    else:
        return ActionRegistry([], [f"手势目录不存在：{source}"])
    errors = []
    groups = []
    for path in files:
        try:
            config = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            errors.append(f"{path.name}: 无法读取——{exc}")
            continue
        if not isinstance(config, dict) or not isinstance(config.get("gestures"), dict):
            errors.append(f"{path.name}: 缺少 gestures 字典，整组拒载")
            continue
        groups.append((path, config))
    accepted = []
    seen = {}  # name -> group file name that registered it
    for path, config in groups:
        conflicts = sorted(n for n in _group_names(config) if n in seen)
        if conflicts:
            errors.append(f"{path.name}: 与 {seen[conflicts[0]]} 重复的动作名 "
                          f"{conflicts}，整组拒载")
            continue
        invalid = _first_recipe_error(path, config)
        if invalid:
            errors.append(f"{path.name}: {invalid}，整组拒载")
            continue
        for name in _group_names(config):
            seen[name] = path.name
        accepted.append((path, config))
    return ActionRegistry(accepted, errors)


def _first_recipe_error(path, config):
    """Run the full recipe_for validation on every gesture and alias."""
    for name in sorted(_group_names(config)):
        try:
            recipe_for(config, name)
        except (ValueError, KeyError, TypeError) as exc:
            return f"动作 {name} 无效：{exc}"
    return None
