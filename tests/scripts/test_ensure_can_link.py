"""The CAN preflight must not touch an already-up interface or arm control."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/ensure_can_link.sh'


class CanLinkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        ip = self.root / 'ip'
        ip.write_text('''#!/bin/sh
if [ "$1" = "-o" ]; then
  if [ -f "$CAN_UP_FLAG" ]; then
    echo '6: can0: <NOARP,UP,LOWER_UP,ECHO> state UP'
  else
    echo '6: can0: <NOARP,ECHO> state DOWN'
  fi
  exit 0
fi
if [ "$1" = "link" ] && [ "$2" = "set" ]; then
  : > "$CAN_UP_FLAG"
  exit 0
fi
exit 2
''')
        ip.chmod(0o755)
        sudo = self.root / 'sudo'
        sudo.write_text('''#!/bin/sh
echo called >> "$SUDO_LOG"
if [ "$1" = "-n" ]; then shift; fi
exec "$@"
''')
        sudo.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.root) + os.pathsep + os.environ['PATH'],
                        CAN_UP_FLAG=str(self.root / 'up'), SUDO_LOG=str(self.root / 'sudo.log'))

    def run_preflight(self):
        return subprocess.run(['bash', str(SCRIPT)], env=self.env,
                              text=True, capture_output=True, check=False)

    def test_already_up_does_not_call_sudo(self):
        (self.root / 'up').touch()
        self.assertEqual(self.run_preflight().returncode, 0)
        self.assertFalse((self.root / 'sudo.log').exists())

    def test_down_is_restored_before_continuing(self):
        result = self.run_preflight()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / 'up').exists())
        self.assertIn('已恢复', result.stderr)


if __name__ == '__main__':
    unittest.main()
