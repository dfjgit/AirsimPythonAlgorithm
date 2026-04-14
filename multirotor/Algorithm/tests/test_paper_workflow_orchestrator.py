import os
import shutil
import sys
import unittest
import uuid
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, call

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import paper_workflow_orchestrator
from paper_workflow_orchestrator import PaperWorkflowOrchestrator


RECOMMENDATIONS = {"ddpg_apf": {"decision": "寤鸿缁", "reasons": ["recent success low"]}}


class PaperWorkflowOrchestratorTests(unittest.TestCase):
    def setUp(self):
        workspace_base = Path.cwd().parents[1]
        self.root = workspace_base / f".tmp_paper_workflow_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=False)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _create_orchestrator(self, command_runner, archive_runner, recommendation_runner):
        return PaperWorkflowOrchestrator(
            workspace_root=self.root,
            command_runner=command_runner,
            archive_runner=archive_runner,
            recommendation_runner=recommendation_runner,
        )

    def test_run_comparison_workflow_records_full_sequence_and_state(self):
        command_runner = Mock(return_value=0)
        archive_runner = Mock()
        recommendation_runner = Mock(return_value=RECOMMENDATIONS)
        orchestrator = self._create_orchestrator(command_runner, archive_runner, recommendation_runner)
        exp_root = orchestrator.create_or_resume_experiment(workflow_type="comparison", alias="main-run")

        orchestrator.run_comparison_workflow(exp_root)

        expected_commands = [
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
        self.assertEqual(state["current_phase"], "stage01_ddpg")
        self.assertEqual(state["steps"]["stage01_ddpg"]["status"], "failed")
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

    def test_main_help_prints_usage_and_exits_cleanly(self):
        stdout = StringIO()

        with self.assertRaises(SystemExit) as ctx, redirect_stdout(stdout):
            paper_workflow_orchestrator.main(["--help"])

        self.assertEqual(ctx.exception.code, 0)
        output = stdout.getvalue().lower()
        self.assertIn("usage:", output)
        self.assertIn("--workflow", output)

    def test_main_comparison_workflow_uses_injected_factory_without_running_real_scripts(self):
        recorded = {}
        expected_exp_root = self.root / "analysis_results" / "workflows" / "comparison" / "cli-run"

        class FakeOrchestrator:
            def create_or_resume_experiment(self, *, workflow_type: str, alias: str = "") -> Path:
                recorded["workflow_type"] = workflow_type
                recorded["alias"] = alias
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
        self.assertEqual(recorded["run_exp_root"], expected_exp_root)
