import json
import os
import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _test_temp_paths import make_temp_dir, suite_temp_root
from paper_workflow_state import (
    create_experiment_root,
    initialize_workflow_state,
    list_resumable_experiments,
    load_workflow_state,
    save_workflow_state,
)


class PaperWorkflowStateTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = suite_temp_root("paper_workflow_state")
        self.tempdir = make_temp_dir("paper_workflow_state")
        self.root = self.tempdir

    def tearDown(self):
        shutil.rmtree(self.tempdir, ignore_errors=True)

    def test_initialize_workflow_state_creates_expected_files(self):
        exp_root = create_experiment_root(
            base_root=self.root,
            workflow_type="comparison",
            alias="main-run",
            now_token="2026-04-14_153000",
        )
        expected_exp_root = self.root / "comparison" / "2026-04-14_153000_main-run"
        self.assertEqual(exp_root, expected_exp_root)
        state = initialize_workflow_state(exp_root, workflow_type="comparison", alias="main-run")
        self.assertEqual(state["workflow_type"], "comparison")
        self.assertEqual(state["status"], "pending")
        self.assertTrue((exp_root / "workflow_state.json").exists())
        self.assertTrue((exp_root / "artifacts").exists())

    def test_initialize_workflow_state_has_coherent_timestamps(self):
        exp_root = create_experiment_root(
            base_root=self.root,
            workflow_type="comparison",
            alias="timestamps",
            now_token="2026-04-14_200000",
        )
        state = initialize_workflow_state(exp_root, workflow_type="comparison", alias="timestamps")
        self.assertEqual(state["created_at"], state["updated_at"])

    def test_save_and_reload_state_round_trip(self):
        exp_root = self.root / "comparison" / "exp-a"
        exp_root.mkdir(parents=True, exist_ok=True)
        state = {"workflow_type": "comparison", "status": "running", "current_phase": "stage01_ddpg"}
        save_workflow_state(exp_root, state)
        loaded = load_workflow_state(exp_root)
        self.assertEqual(loaded["current_phase"], "stage01_ddpg")

    def test_list_resumable_experiments_prefers_non_completed_states(self):
        comparison_root = self.root / "comparison"
        comparison_root.mkdir(parents=True, exist_ok=True)
        exp_a = comparison_root / "2026-04-14_153000_a"
        exp_b = comparison_root / "2026-04-14_160000_b"
        exp_a.mkdir()
        exp_b.mkdir()
        save_workflow_state(exp_a, {"workflow_type": "comparison", "status": "completed", "updated_at": "2026-04-14 15:45:00"})
        save_workflow_state(exp_b, {"workflow_type": "comparison", "status": "interrupted", "updated_at": "2026-04-14 16:10:00"})
        resumable = list_resumable_experiments(self.root, workflow_type="comparison")
        self.assertEqual([item["experiment_root"].name for item in resumable], ["2026-04-14_160000_b"])

    def test_create_experiment_root_sanitizes_and_disambiguates_paths(self):
        alias = "main/run"
        first = create_experiment_root(
            base_root=self.root,
            workflow_type="comparison",
            alias=alias,
            now_token="2026-04-14_153000",
        )
        expected_first = self.root / "comparison" / "2026-04-14_153000_main_run"
        self.assertEqual(first, expected_first)
        second = create_experiment_root(
            base_root=self.root,
            workflow_type="comparison",
            alias=alias,
            now_token="2026-04-14_153000",
        )
        expected_second = self.root / "comparison" / "2026-04-14_153000_main_run_1"
        self.assertEqual(second, expected_second)

    def test_create_experiment_root_rejects_unsafe_workflow_type(self):
        result = create_experiment_root(
            base_root=self.root,
            workflow_type="..",
            alias="run",
            now_token="2026-04-14_180000",
        )
        expected = self.root / "workflow" / "2026-04-14_180000_run"
        self.assertEqual(result, expected)

    def test_save_workflow_state_keeps_updated_at_consistent(self):
        exp_root = self.root / "comparison" / "exp-a"
        exp_root.mkdir(parents=True, exist_ok=True)
        state = {
            "workflow_type": "comparison",
            "status": "running",
            "current_phase": "stage01_ddpg",
            "updated_at": "stale",
        }
        save_workflow_state(exp_root, state)
        loaded = load_workflow_state(exp_root)
        self.assertEqual(state["updated_at"], loaded["updated_at"])
        self.assertNotEqual(state["updated_at"], "stale")

    def test_list_resumable_experiments_skips_malformed_files(self):
        valid_exp = create_experiment_root(
            base_root=self.root,
            workflow_type="comparison",
            alias="valid",
            now_token="2026-04-14_170000",
        )
        initialize_workflow_state(valid_exp, workflow_type="comparison", alias="valid")
        malformed = self.root / "comparison" / "malformed"
        malformed.mkdir(parents=True, exist_ok=True)
        (malformed / "workflow_state.json").write_text("not json", encoding="utf-8")
        (malformed / "artifacts").mkdir(exist_ok=True)
        resumable = list_resumable_experiments(self.root, workflow_type="comparison")
        names = [item["experiment_root"].name for item in resumable]
        self.assertIn(valid_exp.name, names)
        self.assertNotIn("malformed", names)

    def test_list_resumable_experiments_skips_wrong_shape_state(self):
        valid_exp = create_experiment_root(
            base_root=self.root,
            workflow_type="comparison",
            alias="good",
            now_token="2026-04-14_175000",
        )
        initialize_workflow_state(valid_exp, workflow_type="comparison", alias="good")
        bad_shape = self.root / "comparison" / "wrong_shape"
        bad_shape.mkdir(parents=True, exist_ok=True)
        (bad_shape / "workflow_state.json").write_text("[]", encoding="utf-8")
        (bad_shape / "artifacts").mkdir(exist_ok=True)
        resumable = list_resumable_experiments(self.root, workflow_type="comparison")
        names = [item["experiment_root"].name for item in resumable]
        self.assertIn(valid_exp.name, names)
        self.assertNotIn("wrong_shape", names)

    def test_create_experiment_root_sanitizes_now_token(self):
        result = create_experiment_root(
            base_root=self.root,
            workflow_type="comparison",
            alias="run",
            now_token="../../escaped",
        )
        self.assertEqual(result, self.root / "comparison" / "escaped_run")

    def test_list_resumable_experiments_unsafe_workflow_type_uses_default(self):
        exp = create_experiment_root(
            base_root=self.root,
            workflow_type="..",
            alias="unsafe",
            now_token="2026-04-14_190000",
        )
        initialize_workflow_state(exp, workflow_type="..", alias="unsafe")
        results = list_resumable_experiments(self.root, workflow_type="..")
        self.assertTrue(results)
        self.assertEqual(results[0]["experiment_root"].parent, self.root / "workflow")

    def test_list_resumable_experiments_handles_non_string_updated_at(self):
        exp = create_experiment_root(
            base_root=self.root,
            workflow_type="comparison",
            alias="numeric-update",
            now_token="2026-04-14_195000",
        )
        state = initialize_workflow_state(exp, workflow_type="comparison", alias="numeric-update")
        state["updated_at"] = 123
        (exp / "workflow_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        results = list_resumable_experiments(self.root, workflow_type="comparison")
        self.assertIn(exp.name, [item["experiment_root"].name for item in results])

    def test_list_resumable_experiments_invalid_updated_at_does_not_outrank_valid(self):
        valid = create_experiment_root(
            base_root=self.root,
            workflow_type="comparison",
            alias="valid",
            now_token="2026-04-14_205000",
        )
        state_valid = initialize_workflow_state(valid, workflow_type="comparison", alias="valid")
        invalid = create_experiment_root(
            base_root=self.root,
            workflow_type="comparison",
            alias="invalid",
            now_token="2026-04-14_205001",
        )
        state_invalid = initialize_workflow_state(invalid, workflow_type="comparison", alias="invalid")
        state_invalid["updated_at"] = 123
        (invalid / "workflow_state.json").write_text(json.dumps(state_invalid, ensure_ascii=False, indent=2), encoding="utf-8")
        results = list_resumable_experiments(self.root, workflow_type="comparison")
        names = [item["experiment_root"].name for item in results]
        self.assertEqual(names[0], valid.name)
        self.assertIn(invalid.name, names)

    def test_list_resumable_experiments_string_updated_at_malformed_not_outrank(self):
        valid = create_experiment_root(
            base_root=self.root,
            workflow_type="comparison",
            alias="good",
            now_token="2026-04-14_210000",
        )
        initialize_workflow_state(valid, workflow_type="comparison", alias="good")
        malformed = create_experiment_root(
            base_root=self.root,
            workflow_type="comparison",
            alias="bad",
            now_token="2026-04-14_210001",
        )
        state = initialize_workflow_state(malformed, workflow_type="comparison", alias="bad")
        state["updated_at"] = "not-a-date"
        (malformed / "workflow_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        results = list_resumable_experiments(self.root, workflow_type="comparison")
        names = [item["experiment_root"].name for item in results]
        self.assertEqual(names[0], valid.name)

    def test_reserved_workflow_type_falls_back_to_default_bucket(self):
        exp = create_experiment_root(
            base_root=self.root,
            workflow_type="con",
            alias="reserved",
            now_token="2026-04-14_220000",
        )
        initialize_workflow_state(exp, workflow_type="con", alias="reserved")
        results = list_resumable_experiments(self.root, workflow_type="con")
        self.assertTrue(results)
        self.assertEqual(results[0]["experiment_root"].parent, self.root / "workflow")


if __name__ == "__main__":
    unittest.main()
