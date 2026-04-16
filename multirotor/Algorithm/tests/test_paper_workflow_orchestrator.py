import os
import shutil
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, call

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _test_temp_paths import make_temp_dir
import paper_workflow_orchestrator
from paper_workflow_orchestrator import PaperWorkflowOrchestrator
from paper_workflow_state import create_experiment_root, initialize_workflow_state, save_workflow_state


RECOMMENDATIONS = {"ddpg_apf": {"decision": "寤鸿缁", "reasons": ["recent success low"]}}


class PaperWorkflowOrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.root = make_temp_dir("paper_workflow")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _create_orchestrator(self, command_runner, archive_runner, recommendation_runner):
        return PaperWorkflowOrchestrator(
            workspace_root=self.root,
            command_runner=command_runner,
            archive_runner=archive_runner,
            recommendation_runner=recommendation_runner,
        )

    def _create_two_stage_orchestrator(self, command_runner, archive_runner, analysis_runner, recommendation_runner):
        return PaperWorkflowOrchestrator(
            workspace_root=self.root,
            command_runner=command_runner,
            archive_runner=archive_runner,
            recommendation_runner=None,
            two_stage_recommendation_runner=recommendation_runner,
            two_stage_analysis_runner=analysis_runner,
        )

    def test_run_comparison_workflow_records_full_sequence_and_state(self):
        command_runner = Mock(return_value=0)
        archive_runner = Mock()
        recommendation_runner = Mock(return_value=RECOMMENDATIONS)
        orchestrator = self._create_orchestrator(command_runner, archive_runner, recommendation_runner)
        exp_root = orchestrator.create_or_resume_experiment(workflow_type="comparison", alias="main-run")

        orchestrator.run_comparison_workflow(exp_root)

        apf_output_root = exp_root / "artifacts" / "apf_baseline_sim"
        expected_commands = [
            call(
                [
                    "cmd.exe",
                    "/d",
                    "/c",
                    "scripts\\Run_APF_Baseline_Simulation.bat",
                    "--out",
                    str(apf_output_root),
                    "--raw-log-dir",
                    str(apf_output_root / "logs"),
                    "--experiment-id",
                    exp_root.name,
                    "--stage-name",
                    "stage00_apf_baseline",
                    "--stage-index",
                    "0",
                ],
                cwd=self.root,
            ),
            call(["cmd.exe", "/d", "/c", "scripts\\Train_DDPG_Weights_Real_Environment.bat"], cwd=self.root),
            call(["cmd.exe", "/d", "/c", "scripts\\Train_DQN_Movement_Real_Environment.bat"], cwd=self.root),
            call(["cmd.exe", "/d", "/c", "scripts\\Run_Four_Group_Benchmark.bat"], cwd=self.root),
            call(
                ["python", "multirotor\\Algorithm\\visualize_training_data.py", "--auto", "--out", "analysis_results"],
                cwd=self.root,
            ),
        ]
        self.assertEqual(command_runner.call_args_list, expected_commands)

        expected_archives = [
            call(self.root, exp_root, algorithm="ddpg_apf", stage_bucket="stage01"),
            call(self.root, exp_root, algorithm="pure_dqn", stage_bucket="stage01"),
        ]
        self.assertEqual(archive_runner.call_args_list, expected_archives)

        recommendation_runner.assert_called_once()

        state = orchestrator.load_state(exp_root)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["current_phase"], "stage02_decision")
        self.assertEqual(state["recommendations"], RECOMMENDATIONS)
        self.assertEqual(state["steps"]["apf_baseline_sim"]["status"], "completed")
        self.assertEqual(state["steps"]["stage01_ddpg"]["status"], "completed")
        self.assertEqual(state["steps"]["stage01_dqn"]["status"], "completed")
        self.assertEqual(state["steps"]["frozen_benchmark"]["status"], "completed")
        self.assertEqual(state["steps"]["training_comparison"]["status"], "completed")
        self.assertEqual(state["steps"]["stage02_decision"]["status"], "completed")

    def test_command_failure_marks_workflow_as_failed(self):
        command_runner = Mock(side_effect=[1])
        archive_runner = Mock()
        recommendation_runner = Mock(return_value={})
        orchestrator = self._create_orchestrator(command_runner, archive_runner, recommendation_runner)
        exp_root = orchestrator.create_or_resume_experiment(workflow_type="comparison", alias="main-run")

        with self.assertRaises(RuntimeError):
            orchestrator.run_comparison_workflow(exp_root)

        state = orchestrator.load_state(exp_root)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["current_phase"], "apf_baseline_sim")
        self.assertEqual(state["steps"]["apf_baseline_sim"]["status"], "failed")
        archive_runner.assert_not_called()
        recommendation_runner.assert_not_called()

    def test_archive_failure_marks_workflow_as_failed(self):
        command_runner = Mock(return_value=0)
        archive_runner = Mock(side_effect=ValueError("archive failure"))
        recommendation_runner = Mock(return_value=RECOMMENDATIONS)
        orchestrator = self._create_orchestrator(command_runner, archive_runner, recommendation_runner)
        exp_root = orchestrator.create_or_resume_experiment(workflow_type="comparison", alias="main-run")

        with self.assertRaises(ValueError):
            orchestrator.run_comparison_workflow(exp_root)

        state = orchestrator.load_state(exp_root)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["current_phase"], "stage01_ddpg")
        self.assertEqual(state["steps"]["stage01_ddpg"]["status"], "failed")
        recommendation_runner.assert_not_called()

    def test_recommendation_failure_marks_workflow_as_failed(self):
        command_runner = Mock(return_value=0)
        archive_runner = Mock()
        recommendation_runner = Mock(side_effect=RuntimeError("no recommendation"))
        orchestrator = self._create_orchestrator(command_runner, archive_runner, recommendation_runner)
        exp_root = orchestrator.create_or_resume_experiment(workflow_type="comparison", alias="main-run")

        with self.assertRaises(RuntimeError):
            orchestrator.run_comparison_workflow(exp_root)

        state = orchestrator.load_state(exp_root)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["current_phase"], "stage02_decision")
        self.assertEqual(state["steps"]["stage02_decision"]["status"], "failed")

    def test_create_or_resume_experiment_reuses_latest_incomplete_when_requested(self):
        orchestrator = self._create_orchestrator(Mock(), Mock(), Mock())
        workflow_root = self.root / "analysis_results" / "workflows"
        older = create_experiment_root(
            base_root=workflow_root,
            workflow_type="comparison",
            alias="older",
            now_token="2026-04-14_150000",
        )
        newer = create_experiment_root(
            base_root=workflow_root,
            workflow_type="comparison",
            alias="newer",
            now_token="2026-04-14_160000",
        )
        completed = create_experiment_root(
            base_root=workflow_root,
            workflow_type="comparison",
            alias="completed",
            now_token="2026-04-14_170000",
        )
        initialize_workflow_state(older, workflow_type="comparison", alias="older")
        initialize_workflow_state(newer, workflow_type="comparison", alias="newer")
        initialize_workflow_state(completed, workflow_type="comparison", alias="completed")
        save_workflow_state(older, {"workflow_type": "comparison", "status": "failed"}, updated_at="2026-04-14 15:05:00")
        save_workflow_state(newer, {"workflow_type": "comparison", "status": "running"}, updated_at="2026-04-14 16:05:00")
        save_workflow_state(completed, {"workflow_type": "comparison", "status": "completed"}, updated_at="2026-04-14 17:05:00")

        exp_root = orchestrator.create_or_resume_experiment(workflow_type="comparison", resume_latest=True)

        self.assertEqual(exp_root, newer)

    def test_run_comparison_workflow_skips_completed_steps_when_resuming(self):
        command_runner = Mock(return_value=0)
        archive_runner = Mock()
        recommendation_runner = Mock(return_value=RECOMMENDATIONS)
        orchestrator = self._create_orchestrator(command_runner, archive_runner, recommendation_runner)
        exp_root = orchestrator.create_or_resume_experiment(workflow_type="comparison", alias="resume-run")
        state = orchestrator.load_state(exp_root)
        state["status"] = "failed"
        state["current_phase"] = "stage01_dqn"
        state["steps"] = {
            "apf_baseline_sim": {"status": "completed"},
            "stage01_ddpg": {"status": "completed"},
            "stage01_dqn": {"status": "failed", "error": "simulated failure"},
        }
        save_workflow_state(exp_root, state)

        orchestrator.run_comparison_workflow(exp_root)

        expected_commands = [
            call(["cmd.exe", "/d", "/c", "scripts\\Train_DQN_Movement_Real_Environment.bat"], cwd=self.root),
            call(["cmd.exe", "/d", "/c", "scripts\\Run_Four_Group_Benchmark.bat"], cwd=self.root),
            call(
                ["python", "multirotor\\Algorithm\\visualize_training_data.py", "--auto", "--out", "analysis_results"],
                cwd=self.root,
            ),
        ]
        self.assertEqual(command_runner.call_args_list, expected_commands)
        self.assertEqual(
            archive_runner.call_args_list,
            [call(self.root, exp_root, algorithm="pure_dqn", stage_bucket="stage01")],
        )
        recommendation_runner.assert_called_once()
        resumed_state = orchestrator.load_state(exp_root)
        self.assertEqual(resumed_state["status"], "completed")
        self.assertEqual(resumed_state["steps"]["apf_baseline_sim"]["status"], "completed")
        self.assertEqual(resumed_state["steps"]["stage01_ddpg"]["status"], "completed")
        self.assertEqual(resumed_state["steps"]["stage01_dqn"]["status"], "completed")
        self.assertEqual(resumed_state["steps"]["frozen_benchmark"]["status"], "completed")
        self.assertEqual(resumed_state["steps"]["training_comparison"]["status"], "completed")
        self.assertEqual(resumed_state["steps"]["stage02_decision"]["status"], "completed")

    def test_run_virtual_real_two_stage_workflow_updates_state_and_archives_outputs(self):
        command_runner = Mock(return_value=0)
        archive_runner = Mock()
        recommendation_runner = Mock(return_value={"decision": "寤鸿缁х画瀹為淇", "reasons": ["efficiency gain remains large"]})
        two_stage_analysis_runner = Mock(return_value={"summary_csv": self.root / "summary.csv"})

        orchestrator = PaperWorkflowOrchestrator(
            workspace_root=self.root,
            command_runner=command_runner,
            archive_runner=archive_runner,
            recommendation_runner=None,
            two_stage_recommendation_runner=recommendation_runner,
            two_stage_analysis_runner=two_stage_analysis_runner,
        )

        exp_root = orchestrator.create_or_resume_experiment(workflow_type="virtual_real_two_stage", alias="real-a")

        orchestrator.run_virtual_real_two_stage_workflow(exp_root, refine_mode="online")

        state = orchestrator.load_state(exp_root)
        self.assertEqual(state["workflow_type"], "virtual_real_two_stage")
        self.assertEqual(state["current_phase"], "real_weighted_refine_decision")
        self.assertEqual(state["recommendations"]["decision"], "寤鸿缁х画瀹為淇")

    def test_virtual_real_two_stage_online_runs_in_expected_order_with_archives(self):
        command_runner = Mock(return_value=0)
        archive_runner = Mock()
        two_stage_analysis_runner = Mock(return_value={"summary_csv": self.root / "summary.csv"})
        two_stage_recommendation_runner = Mock(return_value={"decision": "ok", "reasons": ["metric"]})

        orchestrator = self._create_two_stage_orchestrator(
            command_runner=command_runner,
            archive_runner=archive_runner,
            analysis_runner=two_stage_analysis_runner,
            recommendation_runner=two_stage_recommendation_runner,
        )
        exp_root = orchestrator.create_or_resume_experiment(workflow_type="virtual_real_two_stage", alias="online-path")

        orchestrator.run_virtual_real_two_stage_workflow(exp_root, refine_mode="online")

        expected_commands = [
            call(["cmd.exe", "/d", "/c", "scripts\\Train_DDPG_Weights_Real_Environment.bat"], cwd=self.root),
            call(["cmd.exe", "/d", "/c", "scripts\\Train_DDPG_Weights_Crazyflie_Online_Single_Episode.bat"], cwd=self.root),
        ]
        self.assertEqual(command_runner.call_args_list, expected_commands)

        expected_archives = [
            call(self.root, exp_root, phase_bucket="sim_pretrain", refine_mode=""),
            call(self.root, exp_root, phase_bucket="real_weighted_refine", refine_mode="online"),
        ]
        self.assertEqual(archive_runner.call_args_list, expected_archives)

        two_stage_analysis_runner.assert_called_once_with(exp_root, refine_mode="online")
        two_stage_recommendation_runner.assert_called_once_with(self.root / "summary.csv")

    def test_virtual_real_two_stage_offline_logs_uses_expected_command_and_archive_args(self):
        command_runner = Mock(return_value=0)
        archive_runner = Mock()
        two_stage_analysis_runner = Mock(return_value={"summary_csv": self.root / "summary.csv"})
        two_stage_recommendation_runner = Mock(return_value={"decision": "ok", "reasons": ["logs"]})

        orchestrator = self._create_two_stage_orchestrator(
            command_runner=command_runner,
            archive_runner=archive_runner,
            analysis_runner=two_stage_analysis_runner,
            recommendation_runner=two_stage_recommendation_runner,
        )
        exp_root = orchestrator.create_or_resume_experiment(workflow_type="virtual_real_two_stage", alias="offline-path")

        orchestrator.run_virtual_real_two_stage_workflow(exp_root, refine_mode="offline_logs")

        expected_commands = [
            call(["cmd.exe", "/d", "/c", "scripts\\Train_DDPG_Weights_Real_Environment.bat"], cwd=self.root),
            call(["cmd.exe", "/d", "/c", "scripts\\Train_DDPG_Weights_Crazyflie_Logs.bat"], cwd=self.root),
        ]
        self.assertEqual(command_runner.call_args_list, expected_commands)

        expected_archives = [
            call(self.root, exp_root, phase_bucket="sim_pretrain", refine_mode=""),
            call(self.root, exp_root, phase_bucket="real_weighted_refine", refine_mode="offline_logs"),
        ]
        self.assertEqual(archive_runner.call_args_list, expected_archives)

        two_stage_analysis_runner.assert_called_once_with(exp_root, refine_mode="offline_logs")
        two_stage_recommendation_runner.assert_called_once_with(self.root / "summary.csv")

    def test_virtual_real_two_stage_rejects_invalid_refine_mode_before_stages(self):
        command_runner = Mock()
        archive_runner = Mock()
        two_stage_analysis_runner = Mock()
        two_stage_recommendation_runner = Mock()

        orchestrator = self._create_two_stage_orchestrator(
            command_runner=command_runner,
            archive_runner=archive_runner,
            analysis_runner=two_stage_analysis_runner,
            recommendation_runner=two_stage_recommendation_runner,
        )
        exp_root = orchestrator.create_or_resume_experiment(workflow_type="virtual_real_two_stage", alias="invalid")

        with self.assertRaises(ValueError):
            orchestrator.run_virtual_real_two_stage_workflow(exp_root, refine_mode="unsupported_mode")

        command_runner.assert_not_called()
        archive_runner.assert_not_called()
        two_stage_analysis_runner.assert_not_called()
        two_stage_recommendation_runner.assert_not_called()
        state = orchestrator.load_state(exp_root)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["current_phase"], "real_weighted_refine")
        self.assertEqual(state["steps"]["real_weighted_refine"]["status"], "failed")

    def test_virtual_real_two_stage_sim_pretrain_command_failure_marks_failed_state(self):
        command_runner = Mock(side_effect=[1])
        archive_runner = Mock()
        two_stage_analysis_runner = Mock()
        two_stage_recommendation_runner = Mock()

        orchestrator = self._create_two_stage_orchestrator(
            command_runner=command_runner,
            archive_runner=archive_runner,
            analysis_runner=two_stage_analysis_runner,
            recommendation_runner=two_stage_recommendation_runner,
        )
        exp_root = orchestrator.create_or_resume_experiment(workflow_type="virtual_real_two_stage", alias="sim-fail")

        with self.assertRaises(RuntimeError):
            orchestrator.run_virtual_real_two_stage_workflow(exp_root, refine_mode="online")

        state = orchestrator.load_state(exp_root)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["current_phase"], "sim_pretrain")
        self.assertEqual(state["steps"]["sim_pretrain"]["status"], "failed")
        archive_runner.assert_not_called()
        two_stage_analysis_runner.assert_not_called()
        two_stage_recommendation_runner.assert_not_called()
        self.assertEqual(command_runner.call_args_list, [call(["cmd.exe", "/d", "/c", "scripts\\Train_DDPG_Weights_Real_Environment.bat"], cwd=self.root)])

    def test_virtual_real_two_stage_real_refine_command_failure_marks_failed_state(self):
        command_runner = Mock(side_effect=[0, 1])
        archive_runner = Mock()
        two_stage_analysis_runner = Mock()
        two_stage_recommendation_runner = Mock()

        orchestrator = self._create_two_stage_orchestrator(
            command_runner=command_runner,
            archive_runner=archive_runner,
            analysis_runner=two_stage_analysis_runner,
            recommendation_runner=two_stage_recommendation_runner,
        )
        exp_root = orchestrator.create_or_resume_experiment(workflow_type="virtual_real_two_stage", alias="refine-fail")

        with self.assertRaises(RuntimeError):
            orchestrator.run_virtual_real_two_stage_workflow(exp_root, refine_mode="online")

        state = orchestrator.load_state(exp_root)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["current_phase"], "real_weighted_refine")
        self.assertEqual(state["steps"]["real_weighted_refine"]["status"], "failed")
        two_stage_analysis_runner.assert_not_called()
        two_stage_recommendation_runner.assert_not_called()
        self.assertEqual(command_runner.call_args_list, [
            call(["cmd.exe", "/d", "/c", "scripts\\Train_DDPG_Weights_Real_Environment.bat"], cwd=self.root),
            call(["cmd.exe", "/d", "/c", "scripts\\Train_DDPG_Weights_Crazyflie_Online_Single_Episode.bat"], cwd=self.root),
        ])
        expected_archives = [
            call(self.root, exp_root, phase_bucket="sim_pretrain", refine_mode=""),
        ]
        self.assertEqual(archive_runner.call_args_list, expected_archives)

    def test_virtual_real_two_stage_analysis_failure_marks_failed_state(self):
        command_runner = Mock(side_effect=[0, 0])
        archive_runner = Mock()
        two_stage_analysis_runner = Mock(side_effect=RuntimeError("analysis fail"))
        two_stage_recommendation_runner = Mock()

        orchestrator = self._create_two_stage_orchestrator(
            command_runner=command_runner,
            archive_runner=archive_runner,
            analysis_runner=two_stage_analysis_runner,
            recommendation_runner=two_stage_recommendation_runner,
        )
        exp_root = orchestrator.create_or_resume_experiment(workflow_type="virtual_real_two_stage", alias="analysis-fail")

        with self.assertRaises(RuntimeError):
            orchestrator.run_virtual_real_two_stage_workflow(exp_root, refine_mode="online")

        state = orchestrator.load_state(exp_root)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["current_phase"], "two_stage_analysis")
        self.assertEqual(state["steps"]["two_stage_analysis"]["status"], "failed")
        two_stage_recommendation_runner.assert_not_called()
        expected_archives = [
            call(self.root, exp_root, phase_bucket="sim_pretrain", refine_mode=""),
            call(self.root, exp_root, phase_bucket="real_weighted_refine", refine_mode="online"),
        ]
        self.assertEqual(archive_runner.call_args_list, expected_archives)

    def test_virtual_real_two_stage_recommendation_failure_marks_failed_state(self):
        command_runner = Mock(side_effect=[0, 0])
        archive_runner = Mock()
        two_stage_analysis_runner = Mock(return_value={"summary_csv": self.root / "summary.csv"})
        two_stage_recommendation_runner = Mock(side_effect=RuntimeError("no rec"))

        orchestrator = self._create_two_stage_orchestrator(
            command_runner=command_runner,
            archive_runner=archive_runner,
            analysis_runner=two_stage_analysis_runner,
            recommendation_runner=two_stage_recommendation_runner,
        )
        exp_root = orchestrator.create_or_resume_experiment(workflow_type="virtual_real_two_stage", alias="recommendation-fail")

        with self.assertRaises(RuntimeError):
            orchestrator.run_virtual_real_two_stage_workflow(exp_root, refine_mode="online")

        state = orchestrator.load_state(exp_root)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["current_phase"], "real_weighted_refine_decision")
        self.assertEqual(state["steps"]["real_weighted_refine_decision"]["status"], "failed")
        expected_archives = [
            call(self.root, exp_root, phase_bucket="sim_pretrain", refine_mode=""),
            call(self.root, exp_root, phase_bucket="real_weighted_refine", refine_mode="online"),
        ]
        self.assertEqual(archive_runner.call_args_list, expected_archives)

    def test_run_virtual_real_two_stage_workflow_skips_completed_steps_when_resuming(self):
        command_runner = Mock(return_value=0)
        archive_runner = Mock()
        two_stage_analysis_runner = Mock(return_value={"summary_csv": self.root / "summary.csv"})
        two_stage_recommendation_runner = Mock(return_value={"decision": "ok", "reasons": ["resume"]})

        orchestrator = self._create_two_stage_orchestrator(
            command_runner=command_runner,
            archive_runner=archive_runner,
            analysis_runner=two_stage_analysis_runner,
            recommendation_runner=two_stage_recommendation_runner,
        )
        exp_root = orchestrator.create_or_resume_experiment(workflow_type="virtual_real_two_stage", alias="resume-two-stage")
        state = orchestrator.load_state(exp_root)
        state["status"] = "failed"
        state["current_phase"] = "real_weighted_refine"
        state["steps"] = {
            "sim_pretrain": {"status": "completed"},
            "real_weighted_refine": {"status": "failed", "error": "simulated failure"},
        }
        save_workflow_state(exp_root, state)

        orchestrator.run_virtual_real_two_stage_workflow(exp_root, refine_mode="online")

        self.assertEqual(
            command_runner.call_args_list,
            [call(["cmd.exe", "/d", "/c", "scripts\\Train_DDPG_Weights_Crazyflie_Online_Single_Episode.bat"], cwd=self.root)],
        )
        self.assertEqual(
            archive_runner.call_args_list,
            [call(self.root, exp_root, phase_bucket="real_weighted_refine", refine_mode="online")],
        )
        two_stage_analysis_runner.assert_called_once_with(exp_root, refine_mode="online")
        two_stage_recommendation_runner.assert_called_once_with(self.root / "summary.csv")
        resumed_state = orchestrator.load_state(exp_root)
        self.assertEqual(resumed_state["status"], "completed")
        self.assertEqual(resumed_state["steps"]["sim_pretrain"]["status"], "completed")
        self.assertEqual(resumed_state["steps"]["real_weighted_refine"]["status"], "completed")
        self.assertEqual(resumed_state["steps"]["two_stage_analysis"]["status"], "completed")
        self.assertEqual(resumed_state["steps"]["real_weighted_refine_decision"]["status"], "completed")
    def test_main_help_prints_usage_and_exits_cleanly(self):
        stdout = StringIO()

        with self.assertRaises(SystemExit) as ctx, redirect_stdout(stdout):
            paper_workflow_orchestrator.main(["--help"])

        self.assertEqual(ctx.exception.code, 0)
        output = stdout.getvalue().lower()
        self.assertIn("usage:", output)
        self.assertIn("--workflow", output)
        self.assertIn("virtual_real_two_stage", output)
        self.assertIn("--refine-mode", output)

    def test_main_comparison_workflow_uses_injected_factory_without_running_real_scripts(self):
        recorded = {}
        expected_exp_root = self.root / "analysis_results" / "workflows" / "comparison" / "cli-run"

        class FakeOrchestrator:
            def create_or_resume_experiment(
                self,
                *,
                workflow_type: str,
                alias: str = "",
                resume_latest: bool = False,
                experiment_root: Path | None = None,
            ) -> Path:
                recorded["workflow_type"] = workflow_type
                recorded["alias"] = alias
                recorded["resume_latest"] = resume_latest
                recorded["experiment_root"] = experiment_root
                return expected_exp_root

            def run_comparison_workflow(self, exp_root: Path) -> None:
                recorded["run_exp_root"] = exp_root

        def orchestrator_factory(*, workspace_root: Path):
            recorded["workspace_root"] = workspace_root
            return FakeOrchestrator()

        exit_code = paper_workflow_orchestrator.main(
            ["--workflow", "comparison", "--workspace-root", str(self.root), "--alias", "cli-run"],
            orchestrator_factory=orchestrator_factory,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(recorded["workspace_root"], self.root)
        self.assertEqual(recorded["workflow_type"], "comparison")
        self.assertEqual(recorded["alias"], "cli-run")
        self.assertFalse(recorded["resume_latest"])
        self.assertIsNone(recorded["experiment_root"])
        self.assertEqual(recorded["run_exp_root"], expected_exp_root)

    def test_main_comparison_workflow_prints_chinese_status_when_ui_lang_is_zh(self):
        recorded = {}
        expected_exp_root = self.root / "analysis_results" / "workflows" / "comparison" / "cli-zh"

        class FakeOrchestrator:
            def create_or_resume_experiment(
                self,
                *,
                workflow_type: str,
                alias: str = "",
                resume_latest: bool = False,
                experiment_root: Path | None = None,
            ) -> Path:
                return expected_exp_root

            def run_comparison_workflow(self, exp_root: Path) -> None:
                recorded["run_exp_root"] = exp_root

        stdout = StringIO()

        def orchestrator_factory(*, workspace_root: Path):
            return FakeOrchestrator()

        original_lang = os.environ.get("AIRSIM_UI_LANG")
        os.environ["AIRSIM_UI_LANG"] = "zh"
        try:
            with redirect_stdout(stdout):
                exit_code = paper_workflow_orchestrator.main(
                    ["--workflow", "comparison", "--workspace-root", str(self.root), "--alias", "cli-zh"],
                    orchestrator_factory=orchestrator_factory,
                )
        finally:
            if original_lang is None:
                os.environ.pop("AIRSIM_UI_LANG", None)
            else:
                os.environ["AIRSIM_UI_LANG"] = original_lang

        self.assertEqual(exit_code, 0)
        self.assertEqual(recorded["run_exp_root"], expected_exp_root)
        output = stdout.getvalue()
        self.assertIn("四组统一仿真对比阶段已启动", output)
        self.assertIn("四组统一仿真对比阶段已完成", output)

    def test_main_virtual_real_two_stage_workflow_defaults_to_online_refine_mode(self):
        recorded = {}
        expected_exp_root = self.root / "analysis_results" / "workflows" / "virtual_real_two_stage" / "cli-two-stage"

        class FakeOrchestrator:
            def create_or_resume_experiment(
                self,
                *,
                workflow_type: str,
                alias: str = "",
                resume_latest: bool = False,
                experiment_root: Path | None = None,
            ) -> Path:
                recorded["workflow_type"] = workflow_type
                recorded["alias"] = alias
                recorded["resume_latest"] = resume_latest
                recorded["experiment_root"] = experiment_root
                return expected_exp_root

            def run_virtual_real_two_stage_workflow(self, exp_root: Path, *, refine_mode: str) -> None:
                recorded["run_exp_root"] = exp_root
                recorded["refine_mode"] = refine_mode

        def orchestrator_factory(*, workspace_root: Path):
            recorded["workspace_root"] = workspace_root
            return FakeOrchestrator()

        exit_code = paper_workflow_orchestrator.main(
            ["--workflow", "virtual_real_two_stage", "--workspace-root", str(self.root), "--alias", "cli-two-stage"],
            orchestrator_factory=orchestrator_factory,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(recorded["workspace_root"], self.root)
        self.assertEqual(recorded["workflow_type"], "virtual_real_two_stage")
        self.assertEqual(recorded["alias"], "cli-two-stage")
        self.assertFalse(recorded["resume_latest"])
        self.assertIsNone(recorded["experiment_root"])
        self.assertEqual(recorded["run_exp_root"], expected_exp_root)
        self.assertEqual(recorded["refine_mode"], "online")

    def test_main_virtual_real_two_stage_workflow_accepts_explicit_refine_mode(self):
        recorded = {}
        expected_exp_root = self.root / "analysis_results" / "workflows" / "virtual_real_two_stage" / "cli-offline"

        class FakeOrchestrator:
            def create_or_resume_experiment(
                self,
                *,
                workflow_type: str,
                alias: str = "",
                resume_latest: bool = False,
                experiment_root: Path | None = None,
            ) -> Path:
                recorded["workflow_type"] = workflow_type
                recorded["alias"] = alias
                recorded["resume_latest"] = resume_latest
                recorded["experiment_root"] = experiment_root
                return expected_exp_root

            def run_virtual_real_two_stage_workflow(self, exp_root: Path, *, refine_mode: str) -> None:
                recorded["run_exp_root"] = exp_root
                recorded["refine_mode"] = refine_mode

        def orchestrator_factory(*, workspace_root: Path):
            recorded["workspace_root"] = workspace_root
            return FakeOrchestrator()

        exit_code = paper_workflow_orchestrator.main(
            [
                "--workflow",
                "virtual_real_two_stage",
                "--refine-mode",
                "offline_logs",
                "--workspace-root",
                str(self.root),
                "--alias",
                "cli-offline",
            ],
            orchestrator_factory=orchestrator_factory,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(recorded["workspace_root"], self.root)
        self.assertEqual(recorded["workflow_type"], "virtual_real_two_stage")
        self.assertEqual(recorded["alias"], "cli-offline")
        self.assertFalse(recorded["resume_latest"])
        self.assertIsNone(recorded["experiment_root"])
        self.assertEqual(recorded["run_exp_root"], expected_exp_root)
        self.assertEqual(recorded["refine_mode"], "offline_logs")

    def test_main_comparison_workflow_can_resume_latest_experiment(self):
        recorded = {}
        expected_exp_root = self.root / "analysis_results" / "workflows" / "comparison" / "resume-run"

        class FakeOrchestrator:
            def create_or_resume_experiment(
                self,
                *,
                workflow_type: str,
                alias: str = "",
                resume_latest: bool = False,
                experiment_root: Path | None = None,
            ) -> Path:
                recorded["workflow_type"] = workflow_type
                recorded["alias"] = alias
                recorded["resume_latest"] = resume_latest
                recorded["experiment_root"] = experiment_root
                return expected_exp_root

            def run_comparison_workflow(self, exp_root: Path) -> None:
                recorded["run_exp_root"] = exp_root

        def orchestrator_factory(*, workspace_root: Path):
            recorded["workspace_root"] = workspace_root
            return FakeOrchestrator()

        exit_code = paper_workflow_orchestrator.main(
            ["--workflow", "comparison", "--workspace-root", str(self.root), "--resume-latest"],
            orchestrator_factory=orchestrator_factory,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(recorded["workspace_root"], self.root)
        self.assertEqual(recorded["workflow_type"], "comparison")
        self.assertTrue(recorded["resume_latest"])
        self.assertIsNone(recorded["experiment_root"])
        self.assertEqual(recorded["run_exp_root"], expected_exp_root)

    def test_main_query_latest_resumable_prints_resume_metadata(self):
        workflow_root = self.root / "analysis_results" / "workflows"
        exp_root = create_experiment_root(
            base_root=workflow_root,
            workflow_type="comparison",
            alias="resume-query",
            now_token="2026-04-14_180000",
        )
        state = initialize_workflow_state(exp_root, workflow_type="comparison", alias="resume-query")
        state["status"] = "running"
        state["current_phase"] = "stage01_ddpg"
        save_workflow_state(exp_root, state, updated_at="2026-04-14 18:30:00")

        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = paper_workflow_orchestrator.main(
                ["--workflow", "comparison", "--workspace-root", str(self.root), "--query-latest-resumable"]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue().strip(), f"{exp_root}|running|stage01_ddpg")
