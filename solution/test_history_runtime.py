"""Agent.act must consume the packaged history without runtime file writes."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import agent as agent_module
import candidates
from test_agent import FakeEnv


class RuntimeHistoryTests(unittest.TestCase):
    def test_act_reads_packaged_history_even_when_raw_csv_has_changed(self):
        env = FakeEnv(cells=1, customers_per_cell=7, ratios=[.5])
        for column in candidates.TARIFF_FEATURES:
            env.tariffs[column] = [1000., 2000.]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data").mkdir()
            source = root / "data" / "change_tariff.csv"
            artifact = root / "historical_candidates.json"
            pd.DataFrame([
                ("t0", "t1", 6000., 9000.),
            ], columns=candidates.HISTORY_COLUMNS).to_csv(source, index=False)
            # Offline preparation is allowed to write the packaged artifact.
            candidates.load_or_build_historical_rank(source, env.tariffs, cache_path=artifact)
            original_artifact = artifact.read_bytes()
            source.write_text(source.read_text() + "t0,t1,7000.0,14000.0\n")
            original_open = Path.open
            writes, raw_reads = [], []

            def audited_open(path, mode="r", *args, **kwargs):
                if any(flag in mode for flag in "wax+"):
                    writes.append((str(path), mode))
                    raise PermissionError("Runtime filesystem is read-only")
                if path == source:
                    raw_reads.append(str(path))
                return original_open(path, mode, *args, **kwargs)

            instance = agent_module.Agent()
            with patch.object(agent_module, "__file__", str(root / "agent.py")), \
                    patch.object(candidates, "HISTORY_PATH", artifact), \
                    patch.object(Path, "open", audited_open):
                rows = instance.act(env)
            self.assertEqual(writes, [], "Agent.act attempted a filesystem write")
            self.assertEqual(raw_reads, [], "Agent.act read raw history instead of the packaged artifact")
            self.assertEqual(artifact.read_bytes(), original_artifact)
            self.assertTrue(rows)
            self.assertTrue(env.requests)
            self.assertNotIn("error", instance.summary)

    def test_missing_or_corrupt_artifact_uses_fallback_without_writes(self):
        for content in (None, "{broken json"):
            with self.subTest(artifact=content), tempfile.TemporaryDirectory() as temporary:
                env = FakeEnv(cells=1, customers_per_cell=7, ratios=[.5])
                for column in candidates.TARIFF_FEATURES:
                    env.tariffs[column] = [1000., 2000.]
                root = Path(temporary)
                (root / "data").mkdir()
                source = root / "data" / "change_tariff.csv"
                artifact = root / "historical_candidates.json"
                pd.DataFrame([("t0", "t1", 6000., 9000.)],
                             columns=candidates.HISTORY_COLUMNS).to_csv(source, index=False)
                if content is not None:
                    artifact.write_text(content)
                original_open = Path.open
                writes, raw_reads = [], []

                def audited_open(path, mode="r", *args, **kwargs):
                    if any(flag in mode for flag in "wax+"):
                        writes.append(str(path))
                        raise PermissionError("Runtime filesystem is read-only")
                    if path == source:
                        raw_reads.append(str(path))
                    return original_open(path, mode, *args, **kwargs)

                instance = agent_module.Agent()
                with patch.object(agent_module, "__file__", str(root / "agent.py")), \
                        patch.object(candidates, "HISTORY_PATH", artifact), \
                        patch.object(Path, "open", audited_open), \
                        self.assertLogs(candidates.LOGGER, level="WARNING"):
                    rows = instance.act(env)
                self.assertEqual(writes, [])
                self.assertEqual(raw_reads, [])
                self.assertTrue(rows)
                self.assertTrue(env.requests)
                self.assertNotIn("error", instance.summary)
                self.assertEqual(artifact.read_text() if artifact.exists() else None, content)


if __name__ == "__main__":
    unittest.main()
