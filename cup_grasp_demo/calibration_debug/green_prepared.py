"""Conservative reuse: identical checked paths, bounded IK seed-pose changes."""
import copy
import json
import numpy as np


def signature(scene, held, margin):
    return json.dumps([scene, held, margin], sort_keys=True,
                      default=lambda value: np.asarray(value).tolist(), allow_nan=False)


def lift_seed_close(old, current, kin, tcp):
    """Bound endpoint reuse only; the new arm path must still be checked."""
    old, current = np.asarray(old), np.asarray(current)
    if old.shape != (7,) or current.shape != (7,) or not np.isfinite([old, current]).all():
        return False
    if np.max(np.abs(old-current)) > np.radians(.5):
        return False
    before, after = kin.forward(old)[0] @ tcp, kin.forward(current)[0] @ tcp
    angle = np.arccos(np.clip((np.trace(before[:3,:3].T @ after[:3,:3])-1)/2, -1, 1))
    return bool(np.linalg.norm(before[:3,3]-after[:3,3]) <= .002 and angle <= np.radians(1.))


class PreparedRoutes:
    def __init__(self):
        self.routes = {}
        self.vertical = {}
        self.events = []

    def put(self, label, plan, scene, held=None, margin=0):
        self.routes[label] = (copy.deepcopy(plan), signature(scene, held, margin))

    def take(self, label, start, targets, scene, held=None, margin=0, *, start_tolerance_deg=0):
        item = self.routes.pop(label, None)
        valid = bool(item and item[1] == signature(scene, held, margin)
                     and np.asarray(start).shape == (7,)
                     and np.isfinite(start).all()
                     and np.max(np.abs(np.asarray(start) - item[0]['start_q_rad'])) <= np.radians(start_tolerance_deg)
                     and np.array_equal(targets, [s['target_q_rad'] for s in item[0]['stages']]))
        self.events.append(dict(label=label, path_reused=valid,
                                reason='start_within_executor_tolerance_same_scene_targets' if valid else 'fresh_path_required'))
        return item[0] if valid else None

    def put_vertical(self, label, start, targets):
        self.vertical[label] = (list(start), copy.deepcopy(targets))

    def take_vertical(self, label, start, kin):
        item = self.vertical.pop(label, None)
        if item is None:
            return None
        old, targets = item
        # Position alone is insufficient: joint branch and orientation must also agree.
        if np.max(np.abs(np.asarray(old) - start)) > np.radians(.02):
            return None
        before, after = kin.forward(old)[0], kin.forward(start)[0]
        if np.linalg.norm(before[:3, 3] - after[:3, 3]) > .0002:
            return None
        return targets
