import os
import shutil
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _test_temp_paths import make_temp_dir
from paper_workflow_archive import (
    archive_comparison_stage_outputs,
    archive_directory_tree,
    archive_two_stage_outputs,
    collect_ddpg_stage_outputs,
)


class PaperWorkflowArchiveTests(unittest.TestCase):
    def setUp(self):
        self.root = make_temp_dir("paper_workflow_archive")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_collect_ddpg_stage_outputs_finds_final_best_and_sidecar(self):
        models_dir = self.root / "multirotor" / "DDPG_Weight" / "models"
        logs_dir = self.root / "multirotor" / "DDPG_Weight" / "airsim_training_logs"
        models_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        final_model_name = "weight_predictor_airsim_20260414_153000.zip"
        stage_meta_name = "weight_predictor_airsim_20260414_153000.stage_meta.json"
        best_model_name = "best_model_20260414_153000.zip"
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
        outputs = collect_ddpg_stage_outputs(self.root, stage_name="stage01")
        self.assertEqual(outputs["final_model"].name, final_stage)
        self.assertEqual(outputs["best_model"].name, best_stage)
        self.assertEqual(outputs["stage_meta"].name, stage_meta_stage)
        self.assertEqual(outputs["training_logs"], [new_log])

    def test_collect_ddpg_stage_outputs_skips_mismatched_best_model(self):
        models_dir = self.root / "multirotor" / "DDPG_Weight" / "models"
        logs_dir = self.root / "multirotor" / "DDPG_Weight" / "airsim_training_logs"
        models_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        stage_token = "20260414_153050"
        final_stage = f"weight_predictor_airsim_{stage_token}.zip"
        stage_meta_stage = f"weight_predictor_airsim_{stage_token}.stage_meta.json"
        best_stage = "best_model_20260414_153100.zip"
        (models_dir / final_stage).write_text("stage final", encoding="utf-8")
        (models_dir / stage_meta_stage).write_text("stage meta", encoding="utf-8")
        (models_dir / best_stage).write_text("stage best", encoding="utf-8")
        log_path = logs_dir / f"ddpg_training_demo_stage01_{stage_token}.csv"
        log_path.write_text("episode,reward\n1,2\n", encoding="utf-8")
        outputs = collect_ddpg_stage_outputs(self.root, stage_name="stage01")
        self.assertEqual(outputs["final_model"].name, final_stage)
        self.assertEqual(outputs["stage_meta"].name, stage_meta_stage)
        self.assertIsNone(outputs["best_model"])
        self.assertEqual(outputs["training_logs"], [log_path])

    def test_collect_ddpg_stage_outputs_without_logs_maintains_run_consistency(self):
        models_dir = self.root / "multirotor" / "DDPG_Weight" / "models"
        logs_dir = self.root / "multirotor" / "DDPG_Weight" / "airsim_training_logs"
        models_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        final_run = "20260414_153000"
        stale_run = "20260415_000000"
        final_model_name = f"weight_predictor_airsim_{final_run}.zip"
        stage_meta_name = f"weight_predictor_airsim_{final_run}.stage_meta.json"
        stale_stage_meta_name = f"weight_predictor_airsim_{stale_run}.stage_meta.json"
        stale_best_model_name = f"best_model_{stale_run}.zip"
        (models_dir / final_model_name).write_text("final run", encoding="utf-8")
        (models_dir / stage_meta_name).write_text("stage meta", encoding="utf-8")
        (models_dir / stale_stage_meta_name).write_text("stale meta", encoding="utf-8")
        (models_dir / stale_best_model_name).write_text("stale best", encoding="utf-8")
        outputs = collect_ddpg_stage_outputs(self.root, stage_name="stage01")
        self.assertEqual(outputs["final_model"].name, final_model_name)
        self.assertEqual(outputs["stage_meta"].name, stage_meta_name)
        self.assertIsNone(outputs["best_model"])

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
        best_model_name = "best_model_20260414_153000.zip"
        log_name = "ddpg_training_demo_stage01_20260414_153000.csv"
        (workspace / "multirotor" / "DDPG_Weight" / "models" / final_model_name).write_text("final", encoding="utf-8")
        (workspace / "multirotor" / "DDPG_Weight" / "models" / stage_meta_name).write_text("{}", encoding="utf-8")
        (workspace / "multirotor" / "DDPG_Weight" / "models" / best_model_name).write_text("best", encoding="utf-8")
        (workspace / "multirotor" / "DDPG_Weight" / "airsim_training_logs" / log_name).write_text("episode,reward\n1,2\n", encoding="utf-8")
        old_log_name = "ddpg_training_demo_stage01_20260413_114500.csv"
        (workspace / "multirotor" / "DDPG_Weight" / "airsim_training_logs" / old_log_name).write_text("episode,reward\n5,6\n", encoding="utf-8")
        archive_comparison_stage_outputs(workspace, exp_root, algorithm="ddpg_apf", stage_bucket="stage01")
        artifact_root = exp_root / "artifacts" / "stage01" / "ddpg_apf"
        self.assertTrue((artifact_root / "models" / final_model_name).exists())
        self.assertTrue((artifact_root / "models" / best_model_name).exists())
        self.assertTrue((artifact_root / "models" / stage_meta_name).exists())
        self.assertTrue((artifact_root / "logs" / log_name).exists())
        self.assertFalse((artifact_root / "logs" / old_log_name).exists())

    def test_archive_comparison_stage_outputs_copies_dqn_model_sidecar_and_stage_logs(self):
        workspace = self.root / "repo"
        exp_root = self.root / "analysis_results" / "workflows" / "comparison" / "exp-1"
        models_dir = workspace / "multirotor" / "DQN_Movement" / "models"
        logs_dir = workspace / "multirotor" / "DQN_Movement" / "logs" / "dqn_scan_data"
        models_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        model_name = "movement_dqn_airsim_final.zip"
        stage_meta_name = "movement_dqn_airsim_final.stage_meta.json"
        stage_token = "comparison_pure_dqn_stage01"
        training_log = f"dqn_training_{stage_token}_20260414_153000.csv"
        scan_log = f"scan_data_{stage_token}_20260414_153000.csv"
        stale_log = "dqn_training_comparison_pure_dqn_stage02_20260414_160000.csv"

        (models_dir / model_name).write_text("model", encoding="utf-8")
        (models_dir / stage_meta_name).write_text(
            '{"experiment_id": "comparison_pure_dqn", "stage_name": "stage01_from_scratch", "stage_index": 1}',
            encoding="utf-8",
        )
        (logs_dir / training_log).write_text("episode,reward\n1,2\n", encoding="utf-8")
        (logs_dir / scan_log).write_text("elapsed_time,scan_ratio\n1,5\n", encoding="utf-8")
        (logs_dir / stale_log).write_text("episode,reward\n9,9\n", encoding="utf-8")

        outputs = archive_comparison_stage_outputs(workspace, exp_root, algorithm="pure_dqn", stage_bucket="stage01")

        artifact_root = exp_root / "artifacts" / "stage01" / "pure_dqn"
        self.assertEqual(outputs["final_model"].name, model_name)
        self.assertEqual(outputs["stage_meta"].name, stage_meta_name)
        self.assertTrue((artifact_root / "models" / model_name).exists())
        self.assertTrue((artifact_root / "models" / stage_meta_name).exists())
        self.assertTrue((artifact_root / "logs" / training_log).exists())
        self.assertTrue((artifact_root / "logs" / scan_log).exists())
        self.assertFalse((artifact_root / "logs" / stale_log).exists())

    def test_archive_two_stage_sim_pretrain_outputs(self):
        workspace = self.root / "repo"
        exp_root = self.root / "analysis_results" / "workflows" / "virtual_real_two_stage" / "exp-1"
        models_dir = workspace / "multirotor" / "DDPG_Weight" / "models"
        logs_dir = workspace / "multirotor" / "DDPG_Weight" / "airsim_training_logs"
        models_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        exp_root.mkdir(parents=True, exist_ok=True)
        model_name = "weight_predictor_airsim_20260414_153000.zip"
        stage_meta_name = "weight_predictor_airsim_20260414_153000.stage_meta.json"
        log_name = "ddpg_training_two_stage_stage01_20260414_153000.csv"
        (models_dir / model_name).write_text("final", encoding="utf-8")
        (models_dir / stage_meta_name).write_text("{}", encoding="utf-8")
        (logs_dir / log_name).write_text("episode,reward\n1,1\n", encoding="utf-8")
        result = archive_two_stage_outputs(workspace, exp_root, phase_bucket="sim_pretrain", refine_mode="")
        artifact_root = exp_root / "artifacts" / "sim_pretrain" / "ddpg_apf"
        self.assertTrue((artifact_root / "models" / model_name).exists())
        self.assertTrue((artifact_root / "models" / stage_meta_name).exists())
        self.assertTrue((artifact_root / "logs" / log_name).exists())
        self.assertEqual(result["copied"]["logs"], [logs_dir / log_name])

    def test_archive_two_stage_sim_pretrain_copies_model_without_logs(self):
        workspace = self.root / "repo"
        exp_root = self.root / "analysis_results" / "workflows" / "virtual_real_two_stage" / "no-logs"
        models_dir = workspace / "multirotor" / "DDPG_Weight" / "models"
        logs_dir = workspace / "multirotor" / "DDPG_Weight" / "airsim_training_logs"
        models_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        exp_root.mkdir(parents=True, exist_ok=True)
        model_name = "weight_predictor_airsim_20260414_153000.zip"
        stage_meta_name = "weight_predictor_airsim_20260414_153000.stage_meta.json"
        (models_dir / model_name).write_text("final", encoding="utf-8")
        (models_dir / stage_meta_name).write_text("{}", encoding="utf-8")
        result = archive_two_stage_outputs(workspace, exp_root, phase_bucket="sim_pretrain", refine_mode="")
        artifact_root = exp_root / "artifacts" / "sim_pretrain" / "ddpg_apf"
        self.assertTrue((artifact_root / "models" / model_name).exists())
        self.assertTrue((artifact_root / "models" / stage_meta_name).exists())
        self.assertEqual(result["copied"]["models"], [models_dir / model_name])
        self.assertEqual(result["copied"]["logs"], [])

    def test_archive_two_stage_sim_pretrain_skips_sidecars_without_final_model(self):
        workspace = self.root / "repo"
        exp_root = self.root / "analysis_results" / "workflows" / "virtual_real_two_stage" / "no-final"
        models_dir = workspace / "multirotor" / "DDPG_Weight" / "models"
        logs_dir = workspace / "multirotor" / "DDPG_Weight" / "airsim_training_logs"
        models_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        exp_root.mkdir(parents=True, exist_ok=True)
        stage_meta_name = "weight_predictor_airsim_20260414_153000.stage_meta.json"
        best_model_name = "best_model_20260414_153000.zip"
        (models_dir / stage_meta_name).write_text("{}", encoding="utf-8")
        (models_dir / best_model_name).write_text("best", encoding="utf-8")
        result = archive_two_stage_outputs(workspace, exp_root, phase_bucket="sim_pretrain", refine_mode="")
        artifact_root = exp_root / "artifacts" / "sim_pretrain" / "ddpg_apf"
        self.assertFalse((artifact_root / "models" / stage_meta_name).exists())
        self.assertFalse((artifact_root / "models" / best_model_name).exists())
        self.assertEqual(result["copied"]["models"], [])
        self.assertEqual(result["copied"]["logs"], [])

    def test_archive_two_stage_online_refine_outputs(self):
        workspace = self.root / "repo"
        exp_root = self.root / "analysis_results" / "workflows" / "virtual_real_two_stage" / "exp-2"
        models_dir = workspace / "multirotor" / "DDPG_Weight" / "models"
        logs_dir = workspace / "multirotor" / "DDPG_Weight" / "crazyflie_logs"
        models_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        exp_root.mkdir(parents=True, exist_ok=True)
        model_name = "weight_predictor_crazyflie_online_20260414_160000.zip"
        (models_dir / model_name).write_text("online", encoding="utf-8")
        online_log_name = "crazyflie_training_online_20260414_160000.csv"
        offline_log_name = "crazyflie_training_logs_20260414_160100.csv"
        (logs_dir / online_log_name).write_text("episode,reward\n1,1\n", encoding="utf-8")
        (logs_dir / offline_log_name).write_text("episode,reward\n2,2\n", encoding="utf-8")
        result = archive_two_stage_outputs(workspace, exp_root, phase_bucket="real_weighted_refine", refine_mode="online")
        artifact_root = exp_root / "artifacts" / "real_weighted_refine" / "online"
        self.assertTrue((artifact_root / "models" / model_name).exists())
        self.assertTrue((artifact_root / "logs" / online_log_name).exists())
        self.assertFalse((artifact_root / "logs" / offline_log_name).exists())
        self.assertEqual(result["copied"]["logs"], [logs_dir / online_log_name])

    def test_archive_two_stage_offline_refine_outputs(self):
        workspace = self.root / "repo"
        exp_root = self.root / "analysis_results" / "workflows" / "virtual_real_two_stage" / "exp-3"
        models_dir = workspace / "multirotor" / "DDPG_Weight" / "models"
        logs_dir = workspace / "multirotor" / "DDPG_Weight" / "crazyflie_logs"
        models_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        exp_root.mkdir(parents=True, exist_ok=True)
        model_name = "weight_predictor_crazyflie_logs_20260414_170000.zip"
        (models_dir / model_name).write_text("offline", encoding="utf-8")
        offline_log_name = "crazyflie_training_logs_20260414_170000.csv"
        online_log_name = "crazyflie_training_online_20260414_170100.csv"
        (logs_dir / offline_log_name).write_text("episode,reward\n1,1\n", encoding="utf-8")
        (logs_dir / online_log_name).write_text("episode,reward\n2,2\n", encoding="utf-8")
        result = archive_two_stage_outputs(workspace, exp_root, phase_bucket="real_weighted_refine", refine_mode="offline_logs")
        artifact_root = exp_root / "artifacts" / "real_weighted_refine" / "offline_logs"
        self.assertTrue((artifact_root / "logs" / offline_log_name).exists())
        self.assertTrue((artifact_root / "models" / model_name).exists())
        self.assertFalse((artifact_root / "logs" / online_log_name).exists())
        self.assertEqual(result["copied"]["logs"], [logs_dir / offline_log_name])
