"""Directory registry: flat merge, per-group defaults, group-level rejection."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.action_registry import load_registry


def gesture(joints=None, hand=None):
    return dict(joints_deg=joints or [0.1, -80.3, -90.4, 110.0, 155.2, -5.0, 5.3],
                hand_0_100=hand or [0, 0, 0, 0, 0, 0])


def group(defaults=None, gestures=None, aliases=None):
    config = dict(gestures=gestures or {})
    config.update(defaults or {})
    if aliases:
        config['aliases'] = aliases
    return config


SPEED_50 = dict(speed_percent=50, finger_duration_s=0.5)
SPEED_30 = dict(speed_percent=30, finger_duration_s=0.5)
SPEED_40 = dict(speed_percent=40, finger_duration_s=0.5)


class RegistryTests(unittest.TestCase):
    def registry(self, files):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        for name, config in files.items():
            (root / name).write_text(json.dumps(config))
        return load_registry(root)

    def test_flat_merge_and_per_group_defaults(self):
        registry = self.registry({
            'a_demo.json': group(SPEED_30, dict(wave=gesture())),
            'b_feedback.json': group(SPEED_50, dict(yeah=gesture()),
                                     aliases=dict(win='yeah')),
        })
        self.assertEqual(sorted(registry.names()), ['wave', 'win', 'yeah'])
        self.assertEqual(registry.errors, [])
        self.assertEqual(registry.recipe('wave')['speed_percent'], 30)
        self.assertEqual(registry.recipe('win')['speed_percent'], 50)

    def test_duplicate_name_rejects_whole_later_file(self):
        registry = self.registry({
            'a_keep.json': group(SPEED_50, dict(yeah=gesture(), wave=gesture())),
            'b_dupe.json': group(SPEED_50, dict(yeah=gesture(), bow=gesture())),
        })
        self.assertEqual(sorted(registry.names()), ['wave', 'yeah'])
        self.assertEqual(len(registry.errors), 1)
        self.assertIn('a_keep.json', registry.errors[0])
        self.assertIn('b_dupe.json', registry.errors[0])
        self.assertIn('yeah', registry.errors[0])
        with self.assertRaises(ValueError):
            registry.recipe('bow')

    def test_invalid_recipe_rejects_only_that_file(self):
        bad = group(SPEED_50, dict(bow=dict(joints_deg=[1] * 6, hand_0_100=[0] * 6)))
        good = group(SPEED_50, dict(wave=gesture()))
        registry = self.registry({'a_bad.json': bad, 'b_good.json': good})
        self.assertEqual(sorted(registry.names()), ['wave'])
        self.assertEqual(len(registry.errors), 1)
        self.assertIn('a_bad.json', registry.errors[0])
        self.assertIn('bow', registry.errors[0])

    def test_alias_pointing_outside_its_file_is_rejected(self):
        lonely = group(SPEED_30, aliases=dict(win='yeah'))
        keeper = group(SPEED_50, dict(yeah=gesture()))
        registry = self.registry({'a_lonely.json': lonely, 'b_keeper.json': keeper})
        self.assertEqual(sorted(registry.names()), ['yeah'])
        self.assertIn('win', registry.errors[0])

    def test_single_file_source_and_missing_directory(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'one.json'
            path.write_text(json.dumps(group(SPEED_40, dict(wave=gesture()))))
            registry = load_registry(path)
            self.assertEqual(registry.names(), ['wave'])
            self.assertEqual(registry.recipe('wave')['speed_percent'], 40)
            missing = load_registry(Path(name) / 'nope')
            self.assertEqual(missing.names(), [])
            self.assertEqual(len(missing.errors), 1)

    def test_shipped_directory_registers_feedback_gestures(self):
        from scripts.action_registry import GESTURES_DIR
        registry = load_registry(GESTURES_DIR)
        self.assertIn('yeah', registry.names())
        self.assertIn('win', registry.names())
        self.assertIn('home', registry.names())
        self.assertEqual(registry.errors, [])


if __name__ == '__main__':
    unittest.main()
