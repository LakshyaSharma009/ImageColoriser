import tempfile
import unittest
from pathlib import Path

from colorizer import history


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "history.db"
        self.output_dir = Path(self.tmp_dir.name) / "outputs"
        history.init_db(self.db_path)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_record_and_list_round_trip(self):
        run_id = history.record_run(
            filename="photo.jpg",
            model_id="vibrant",
            saturation=1.0,
            width=100,
            height=80,
            elapsed_s=0.5,
            output_bytes=b"fake-png-bytes",
            db_path=self.db_path,
            output_dir=self.output_dir,
        )
        self.assertIsInstance(run_id, int)

        runs = history.list_runs(db_path=self.db_path)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["filename"], "photo.jpg")
        self.assertEqual(runs[0]["model_id"], "vibrant")
        self.assertTrue(Path(runs[0]["output_path"]).exists())

    def test_list_runs_orders_most_recent_first(self):
        for name in ("a.jpg", "b.jpg", "c.jpg"):
            history.record_run(
                filename=name,
                model_id="vibrant",
                saturation=1.0,
                width=10,
                height=10,
                elapsed_s=0.1,
                output_bytes=b"x",
                db_path=self.db_path,
                output_dir=self.output_dir,
            )
        runs = history.list_runs(db_path=self.db_path)
        self.assertEqual([r["filename"] for r in runs], ["c.jpg", "b.jpg", "a.jpg"])

    def test_clear_runs_empties_table(self):
        history.record_run(
            filename="photo.jpg",
            model_id="vibrant",
            saturation=1.0,
            width=10,
            height=10,
            elapsed_s=0.1,
            output_bytes=b"x",
            db_path=self.db_path,
            output_dir=self.output_dir,
        )
        history.clear_runs(db_path=self.db_path)
        self.assertEqual(history.list_runs(db_path=self.db_path), [])

    def test_prune_caps_history_size_and_removes_output_files(self):
        paths = []
        for i in range(5):
            history.record_run(
                filename=f"{i}.jpg",
                model_id="vibrant",
                saturation=1.0,
                width=10,
                height=10,
                elapsed_s=0.1,
                output_bytes=b"x",
                db_path=self.db_path,
                output_dir=self.output_dir,
            )
        for run in history.list_runs(db_path=self.db_path):
            paths.append(Path(run["output_path"]))

        history._prune(self.db_path, max_entries=2)

        remaining = history.list_runs(db_path=self.db_path)
        self.assertEqual(len(remaining), 2)
        self.assertEqual([r["filename"] for r in remaining], ["4.jpg", "3.jpg"])

        surviving_paths = {Path(r["output_path"]) for r in remaining}
        for path in paths:
            if path in surviving_paths:
                self.assertTrue(path.exists())
            else:
                self.assertFalse(path.exists())

    def test_repeated_calls_do_not_leak_or_error_across_many_operations(self):
        for i in range(20):
            history.record_run(
                filename=f"stress-{i}.jpg",
                model_id="natural",
                saturation=1.2,
                width=5,
                height=5,
                elapsed_s=0.01,
                output_bytes=b"y",
                db_path=self.db_path,
                output_dir=self.output_dir,
            )
            history.list_runs(db_path=self.db_path)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "history.db"
        history.init_db(self.db_path)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _record(self, **overrides):
        kwargs = dict(
            model="vibrant",
            dataset="sample",
            image_count=10,
            saturation=1.0,
            device="CPU",
            psnr_mean=28.5,
            ssim_mean=0.91,
            lpips_mean=None,
            latency_mean_seconds=0.2,
            latency_median_seconds=0.18,
            latency_p95_seconds=0.3,
            git_commit="abc1234",
            db_path=self.db_path,
        )
        kwargs.update(overrides)
        return history.record_experiment(**kwargs)

    def test_record_and_list_round_trip(self):
        experiment_id = self._record()
        self.assertIsInstance(experiment_id, int)

        experiments = history.list_experiments(db_path=self.db_path)
        self.assertEqual(len(experiments), 1)
        self.assertEqual(experiments[0]["model"], "vibrant")
        self.assertEqual(experiments[0]["dataset"], "sample")
        self.assertEqual(experiments[0]["git_commit"], "abc1234")

    def test_list_experiments_orders_most_recent_first(self):
        for dataset in ("a", "b", "c"):
            self._record(dataset=dataset)
        experiments = history.list_experiments(db_path=self.db_path)
        self.assertEqual([e["dataset"] for e in experiments], ["c", "b", "a"])

    def test_get_experiment_returns_matching_row(self):
        experiment_id = self._record(dataset="lookup-me")
        experiment = history.get_experiment(experiment_id, db_path=self.db_path)
        self.assertIsNotNone(experiment)
        self.assertEqual(experiment["dataset"], "lookup-me")

    def test_get_experiment_returns_none_for_missing_id(self):
        self.assertIsNone(history.get_experiment(999999, db_path=self.db_path))

    def test_clear_experiments_empties_table(self):
        self._record()
        history.clear_experiments(db_path=self.db_path)
        self.assertEqual(history.list_experiments(db_path=self.db_path), [])

    def test_git_commit_is_auto_populated_when_not_given(self):
        experiment_id = self._record(git_commit=None)
        experiment = history.get_experiment(experiment_id, db_path=self.db_path)
        # Either a short hash (inside a git checkout) or None (outside one) -- never crashes either way.
        self.assertTrue(experiment["git_commit"] is None or isinstance(experiment["git_commit"], str))

    def test_experiments_table_coexists_with_runs_table(self):
        """init_db() must keep working against a database that already has a `runs` table."""
        history.record_run(
            filename="photo.jpg",
            model_id="vibrant",
            saturation=1.0,
            width=10,
            height=10,
            elapsed_s=0.1,
            output_bytes=b"x",
            db_path=self.db_path,
            output_dir=Path(self.tmp_dir.name) / "outputs",
        )
        # Re-running init_db() against a DB that already has `runs` must not error,
        # and must still create `experiments`.
        history.init_db(self.db_path)
        self._record()

        self.assertEqual(len(history.list_runs(db_path=self.db_path)), 1)
        self.assertEqual(len(history.list_experiments(db_path=self.db_path)), 1)


if __name__ == "__main__":
    unittest.main()
