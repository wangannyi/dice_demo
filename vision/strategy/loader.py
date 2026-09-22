"""Grasp strategy loader; one JSON per object type, pipeline reads via this API."""
import json
from pathlib import Path
from types import SimpleNamespace

STRATEGY_DIR = Path(__file__).resolve().parent


def load_strategy(name_or_path):
    """Load a strategy JSON by name (e.g. 'green_cup') or relative path."""
    if isinstance(name_or_path, (str, Path)) and str(name_or_path).endswith(".json"):
        path = Path(name_or_path)
        if not path.is_absolute():
            path = STRATEGY_DIR.parent.parent / path
    else:
        path = STRATEGY_DIR / f"{name_or_path}.json"
    if not path.is_file():
        raise ValueError(f"策略文件不存在: {path}")
    raw = json.loads(path.read_text())
    required = ("contact_offset_base_mm", "grip_targets_0_100",
                "open_targets_0_100", "lift_mm")
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"策略缺少必需字段: {missing}")
    return SimpleNamespace(**raw)
