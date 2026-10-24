"""A late API entry cannot hold the inferred pageable-download slot."""
import importlib.util
from pathlib import Path
import unittest

source = Path(__file__).resolve().parents[1] / 'tools/research/optimizer_d2h_review_20261023/d2h_lock_chain.py'
spec = importlib.util.spec_from_file_location('d2h_wait_attribution', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class WaitAttributionTest(unittest.TestCase):
    def test_api_entered_after_kernel_has_no_hold(self):
        self.assertEqual(module.waiting_interval(10, 200, 100), (200, 200))

    def test_idle_gap_before_api_entry_is_excluded(self):
        self.assertEqual(module.waiting_interval(10, 60, 100), (60, 100))

    def test_queued_call_does_not_own_previous_download(self):
        self.assertEqual(module.waiting_interval(80, 20, 100), (80, 100))

    def test_completed_kernel_waiting_behind_previous_call(self):
        self.assertEqual(module.waiting_interval(120, 20, 100), (120, 120))


if __name__ == '__main__':
    unittest.main()
