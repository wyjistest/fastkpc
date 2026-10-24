"""Hand-computed cases for counterfactual scheduling and overlapping time."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import interval_stats, schedule


class AttributionTests(unittest.TestCase):
    def test_barrier_removal_preserves_original_order(self):
        # Two workers: [4,1] then [1,1] takes 5 with a barrier, 4 with refill.
        self.assertEqual(schedule([4, 1], 2) + schedule([1, 1], 2), 5)
        self.assertEqual(schedule([4, 1, 1, 1], 2), 4)

    def test_long_task_is_irreducible(self):
        # Many slots cannot shorten the longest original request.
        self.assertEqual(schedule([1, 1, 10, 1], 32), 10)

    def test_overlap_is_union_not_worker_sum(self):
        s = 10**9
        r = interval_stats([(0, 3*s), (s, 4*s)], 0, 5*s)
        self.assertEqual(r['union_sec'], 4)
        self.assertEqual(r['no_active_sec'], 1)
        self.assertEqual(r['peak'], 2)
        self.assertAlmostEqual(r['mean_active'], 6/5)

    def test_clipped_tail_and_touching_intervals(self):
        s = 10**9
        r = interval_stats([(0, 2*s), (2*s, 5*s)], s, 4*s)
        self.assertEqual(r['union_sec'], 3)
        self.assertEqual(r['no_active_sec'], 0)
        self.assertEqual(r['peak'], 1)


if __name__ == '__main__':
    unittest.main()
