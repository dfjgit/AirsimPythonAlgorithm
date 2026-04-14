import os
import shutil
import sys
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from paper_workflow_archive import (
    archive_comparison_stage_outputs,
    archive_directory_tree,
    collect_ddpg_stage_outputs,
)


class PaperWorkflowArchiveTests(unittest.TestCase):
    def setUp(self):
        base = Path(os.getcwd()) / "pwf_tmp"
        base.mkdir(exist_ok=True)
        self.root = base / uuid.uuid4().hex[:8]
        self.root.mkdir(parents=True, exist_ok=False)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_collect_ddpg_stage_outputs_finds_final_best_and_sidecar(self):
        models_dir = self.root / "multirotor" / "DDPG_Weight" / "models"
        logs_dir = self.root / "multirotor" / "DDPG_Weight" / "airsim_training_logs"
        models_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        final_model_name = "weight_predictor_airsim_20260414_153000.zip"
        stage_meta_name = "weight_predictor_airsim_20260414_153000.stage_meta.json"
        best_model_name = "best_model_20260414_153001.zip"
        log_name = "ddpg_training_demo_stage01_20260414_153000.csv"
        (models_dir / final_model_name).write_text("final", encoding="utf-8")
        (models_dir / stage_meta_name).write_text("{}", encoding="utf-8")
        (models_dir / best_model_name).write_text("best", encoding="utf-8")
        (logs_dir / log_name).write_text("episode,reward\n1,2\n", encoding="utf-8")
        outputs = collect_ddpg_stage_outputs(self.root, stage_name="stage01")
        self.assertTrue(outputs["final_model"].name.endswith(".zip"))
        self.assertTrue(outputs["stage_meta"].name.endswith(".stage_meta.json"))
        self.assertIn("training_logs", outputs)
        self.assertEqual(outputs["best_model"].name, best_model_name)
        self.assertIn(logs_dir / log_name, outputs["training_logs"])

    def test_collect_ddpg_stage_outputs_uses_stage_run_token(self):
        models_dir = self.root / "multirotor" / "DDPG_Weight" / "models"
        logs_dir = self.root / "multirotor" / "DDPG_Weight" / "airsim_training_logs"
        models_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        stage_run = "20260414_153050"
        older_run = "20260414_153000"
        stage_prefix = "weight_predictor_airsim"
        final_stage = f"{stage_prefix}_{stage_run}.zip"
        stage_meta_stage = f"{stage_prefix}_{stage_run}.stage_meta.json"
        best_stage = f"best_model_{stage_run}.zip"
        # create stage-specific artifacts first so later stale files exercise token filtering
        (models_dir / final_stage).write_text("stage final", encoding="utf-8")
        (models_dir / stage_meta_stage).write_text("stage meta", encoding="utf-8")
        (models_dir / best_stage).write_text("stage best", encoding="utf-8")
        stale_final = f"{stage_prefix}_20260415_000000.zip"
        stale_meta = f"{stage_prefix}_20260415_000000.stage_meta.json"
        stale_best = "best_model_20260415_000000.zip"
        (models_dir / stale_final).write_text("stale final", encoding="utf-8")
        (models_dir / stale_meta).write_text("stale meta", encoding="utf-8")
        (models_dir / stale_best).write_text("stale best", encoding="utf-8")
        old_log = logs_dir / f"ddpg_training_demo_stage01_{older_run}.csv"
        old_log.write_text("episode,reward\n1,2\n", encoding="utf-8")
        new_log = logs_dir / f"ddpg_training_demo_stage01_{stage_run}.csv"
        new_log.write_text("episode,reward\n3,4\n", encoding="utf-8")
        timestamp = 1_706_000_000
        os.utime(old_log, (timestamp, timestamp))
        os.utime(new_log, (timestamp + 10, timestamp + 10))
        outputs = collect_ddpg_stage_outputs(self.root, stage_name="stage01")
        self.assertEqual(outputs["final_model"].name, final_stage)
        self.assertEqual(outputs["best_model"].name, best_stage)
        self.assertEqual(outputs["stage_meta"].name, stage_meta_stage)
        self.assertEqual(outputs["training_logs"][0], new_log)
        self.assertEqual(outputs["training_logs"][1], old_log)

    def test_archive_directory_tree_copies_existing_tree(self):
        src = self.root / "src"
        dst = self.root / "dst"
        src.mkdir()
        (src / "summary.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        archive_directory_tree(src, dst)
        self.assertTrue((dst / "summary.csv").exists())

    def test_archive_comparison_stage_outputs_copies_models_and_logs(self):
        workspace = self.root / "repo"
        exp_root = self.root / "analysis_results" / "workflows" / "comparison" / "exp-1"
        (workspace / "multirotor" / "DDPG_Weight" / "models").mkdir(parents=True, exist_ok=True)
        (workspace / "multirotor" / "DDPG_Weight" / "airsim_training_logs").mkdir(parents=True, exist_ok=True)
        final_model_name = "weight_predictor_airsim_20260414_153000.zip"
        stage_meta_name = "weight_predictor_airsim_20260414_153000.stage_meta.json"
        best_model_name = "best_model_20260414_153001.zip"
        log_name = "ddpg_training_demo_stage01_20260414_153000.csv"
        (workspace / "multirotor" / "DDPG_Weight" / "models" / final_model_name).write_text("final", encoding="utf-8")
        (workspace / "multirotor" / "DDPG_Weight" / "models" / stage_meta_name).write_text("{}", encoding="utf-8")
        (workspace / "multirotor" / "DDPG_Weight" / "models" / best_model_name).write_text("best", encoding="utf-8")
        (workspace / "multirotor" / "DDPG_Weight" / "airsim_training_logs" / log_name).write_text("episode,reward\n1,2\n", encoding="utf-8")
        archive_comparison_stage_outputs(workspace, exp_root, algorithm="ddpg_apf", stage_bucket="stage01")
        artifact_root = exp_root / "artifacts" / "stage01" / "ddpg_apf"
        self.assertTrue((artifact_root / "models" / final_model_name).exists())
        self.assertTrue((artifact_root / "models" / best_model_name).exists())
        self.assertTrue((artifact_root / "models" / stage_meta_name).exists())
        self.assertTrue((artifact_root / "logs" / log_name).exists())
