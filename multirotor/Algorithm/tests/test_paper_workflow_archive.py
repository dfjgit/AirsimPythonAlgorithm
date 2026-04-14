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
        (models_dir / "weight_predictor_airsim_20260414_153000.zip").write_text("final", encoding="utf-8")
        (models_dir / "weight_predictor_airsim_20260414_153000.stage_meta.json").write_text("{}", encoding="utf-8")
        (models_dir / "best_model_20260414_153001.zip").write_text("best", encoding="utf-8")
        (logs_dir / "ddpg_training_demo_stage01_20260414_153000.csv").write_text("episode,reward\n1,2\n", encoding="utf-8")
        outputs = collect_ddpg_stage_outputs(self.root, stage_name="stage01")
        self.assertTrue(outputs["final_model"].name.endswith(".zip"))
        self.assertTrue(outputs["stage_meta"].name.endswith(".stage_meta.json"))
        self.assertIn("training_logs", outputs)

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
        (workspace / "multirotor" / "DDPG_Weight" / "models" / "weight_predictor_airsim_20260414_153000.zip").write_text("final", encoding="utf-8")
        (workspace / "multirotor" / "DDPG_Weight" / "models" / "weight_predictor_airsim_20260414_153000.stage_meta.json").write_text("{}", encoding="utf-8")
        (workspace / "multirotor" / "DDPG_Weight" / "airsim_training_logs" / "ddpg_training_demo_stage01_20260414_153000.csv").write_text("episode,reward\n1,2\n", encoding="utf-8")
        archive_comparison_stage_outputs(workspace, exp_root, algorithm="ddpg_apf", stage_bucket="stage01")
        self.assertTrue((exp_root / "artifacts" / "stage01" / "ddpg_apf" / "models").exists())
        self.assertTrue((exp_root / "artifacts" / "stage01" / "ddpg_apf" / "logs").exists())
