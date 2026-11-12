"""Host-only failure-path checks for evidence portability and decision gates."""
import json
from pathlib import Path
import tempfile
import unittest

import reproduce as repro


class EvidenceTests(unittest.TestCase):
    def test_relocation_and_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "relocated evidence"
            root.mkdir()
            (root / "receipt").write_bytes(b"frozen evidence")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"evidence": {"receipt": repro.sha(root / "receipt")}}))
            anchor = repro.sha(manifest)
            self.assertEqual(repro.verify_manifest(root, manifest, anchor), 1)
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                repro.verify_manifest(root, manifest, "0" * 64)
            (root / "receipt").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "Evidence mismatch"):
                repro.verify_manifest(root, manifest, anchor)
            (root / "receipt").unlink()
            with self.assertRaisesRegex(RuntimeError, "Missing/escaping"):
                repro.verify_manifest(root, manifest, anchor)

    def test_paths_cannot_escape_evidence_root(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "evidence"
            root.mkdir()
            outside = parent / "outside"
            outside.write_bytes(b"outside")
            (root / "link").symlink_to(outside)
            for name in ("../outside", str(outside), "link"):
                with self.subTest(name=name), self.assertRaises(RuntimeError):
                    repro.relative_file(root, name)

    def test_empty_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text('{"evidence": {}}')
            with self.assertRaisesRegex(RuntimeError, "Empty"):
                repro.verify_manifest(root, manifest)


class ProtocolTests(unittest.TestCase):
    @staticmethod
    def runs(off=(100, 100, 100), on=(95, 95, 95)):
        runs = {}
        for prefix, mode, times in (("A", "off", off), ("C", "on", on)):
            for i, seconds in enumerate(times, 1):
                runs[f"{prefix}{i}"] = dict(mode=mode, outer_sec=seconds,
                                           skeleton_sec=seconds * .4, orientation_sec=seconds * .6)
        return runs

    def test_exact_speed_and_nonregression_boundaries(self):
        self.assertEqual(repro.performance(self.runs())["speed"], "PROVEN")
        boundary = repro.performance(self.runs(on=(105, 105, 105)))
        self.assertTrue(boundary["non_regression"])
        self.assertEqual(boundary["speed"], "NOT_PROVEN")
        self.assertEqual(repro.performance(self.runs(on=(105.01,) * 3))["speed"], "REGRESSION")

    def test_median_gain_alone_does_not_prove_speed(self):
        result = repro.performance(self.runs(off=(90, 100, 110), on=(89, 90, 91)))
        self.assertLessEqual(result["ratio"], .95)
        self.assertEqual(result["speed"], "NOT_PROVEN")

    def test_invalid_or_mislabelled_timings_rejected(self):
        for field in ("outer_sec", "skeleton_sec", "orientation_sec"):
            for value in (None, float("nan"), float("inf"), -1, True):
                runs = self.runs()
                runs["C3"][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(RuntimeError):
                    repro.performance(runs)
        runs = self.runs()
        runs["C1"]["mode"] = "off"
        with self.assertRaises(RuntimeError):
            repro.performance(runs)
        del runs["C1"]
        with self.assertRaises(RuntimeError):
            repro.performance(runs)

    def test_affinity_is_explicit_and_bounded(self):
        self.assertEqual(repro.cpus("0-2,4", {0, 1, 2, 4}), [0, 1, 2, 4])
        self.assertEqual(repro.cpus(None, {4, 6}), [4, 6])
        for value in ("", "-1", "2-1", "0-100000000", "0,,1", "3", "0-1-2"):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                repro.cpus(value, {0, 1, 2, 4})


if __name__ == "__main__":
    unittest.main()
