"""Terminal aliases preserve the resident control protocol."""

import json
import unittest

from cup_grasp_demo.flow.control_commands import parse_command


class TerminalCommandsTest(unittest.TestCase):
    def test_aliases(self):
        pairs = {
            'next': dict(command='advance'),
            'n': dict(command='advance'),
            'advance': dict(command='advance'),
            'run': dict(command='advance', until='RETURN_HOME'),
            'g': dict(command='advance', until='RETURN_HOME'),
            'until lift': dict(command='advance', until='LIFT'),
            'home': dict(command='action', name='home'),
            'action rps-ready': dict(command='action', name='rps-ready'),
            'quit': dict(command='close'),
            'q': dict(command='close'),
            '?': dict(command='help'),
        }
        for name in ('status', 'actions', 'reload', 'refresh_perception',
                     'query_pose', 'close', 'help'):
            pairs[name] = dict(command=name)
        for text, expected in pairs.items():
            with self.subTest(text=text):
                self.assertEqual(parse_command(text), expected)
        self.assertEqual(parse_command('  UNTIL  grip \n'),
                         dict(command='advance', until='GRIP'))

    def test_json_is_preserved_including_invalid_request_shapes(self):
        for request in (dict(command='advance', id='test', rounds=3), None,
                        [], 2, True, 'run'):
            with self.subTest(request=request):
                self.assertEqual(parse_command(json.dumps(request)), request)
        with self.assertRaises(json.JSONDecodeError):
            parse_command('{"command":')

    def test_incomplete_or_extra_arguments_are_rejected(self):
        for text in ('run twice', 'next home', 'until', 'action', 'action rock paper',
                     'home now', 'ru', 'stop'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_command(text)
        self.assertIsNone(parse_command(' \n'))
