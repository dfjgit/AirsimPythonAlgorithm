import os
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, call

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
