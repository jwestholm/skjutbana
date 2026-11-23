"""PANG live launcher sync preflight; subprocess is mocked, no hardware."""
import unittest
from unittest.mock import patch
from automation import play_light_targets


class BoardSyncLauncherTests(unittest.TestCase):
    def test_sync_uses_the_existing_backed_up_calibration_check(self):
        with patch('automation.play_light_targets.subprocess.run') as run:
            run.return_value.returncode = 0
            self.assertEqual(play_light_targets.sync_board(), 0)
        args, kwargs = run.call_args
        self.assertEqual(args[0][1:], ['-m','automation.bench','calibration-check','--save'])
        self.assertEqual(kwargs['cwd'], play_light_targets.REPO)

    def test_failed_sync_stops_before_game_start(self):
        with patch('automation.play_light_targets.sync_board', return_value=2), \
             patch('automation.play_light_targets.os.environ', {}) as env:
            with self.assertRaisesRegex(SystemExit, 'game not started'):
                play_light_targets.main(['--object-highlight'])
            self.assertNotIn('SKJUTBANA_PANG_OBJECT_HIGHLIGHT', env)

    def test_running_api_rejects_client_side_sync_request(self):
        with patch('automation.play_light_targets.sync_board') as sync:
            with self.assertRaises(SystemExit):
                play_light_targets.main(['--api','--object-highlight'])
            sync.assert_not_called()


if __name__ == '__main__':
    unittest.main()
