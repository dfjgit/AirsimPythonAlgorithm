import json
import os
import shutil
import subprocess
import unittest
import uuid
from pathlib import Path


@unittest.skipUnless(os.name == "nt", "start_en.bat is Windows-only")
class StartEnBatTests(unittest.TestCase):
    def _run_start_en_bat(self, input_text="0\r\n", extra_env=None):
        script = Path(__file__).resolve().parent / "start_en.bat"
        env = dict(os.environ)
        env["AIRSIM_TEST_NO_PAUSE"] = "1"
        env["AIRSIM_TEST_EXIT_AFTER_TOGGLE"] = "1"
        if extra_env:
            env.update(extra_env)
        completed = subprocess.run(
            ["cmd.exe", "/d", "/c", str(script)],
            input=input_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(script.parent),
            env=env,
            timeout=8,
        )
        combined_output = f"{completed.stdout}\n{completed.stderr}"
        return completed, combined_output

    def test_start_en_bat_menu_exits_without_cmd_parse_errors(self):
        completed, combined_output = self._run_start_en_bat()
        self.assertEqual(completed.returncode, 0, msg=combined_output)
        self.assertIn("AirSim", combined_output)
        self.assertIn("Console", combined_output)
        self.assertNotIn(
            "not recognized as an internal or external command", combined_output
        )

    def test_start_en_bat_menu_uses_formal_console_copy(self):
        _, combined_output = self._run_start_en_bat()
        self.assertIn("AirSim UAV Simulation Platform - Console", combined_output)
        self.assertIn("=== Analysis ===", combined_output)
        self.assertIn("[C] Clean Up Training and Analysis Outputs", combined_output)
        self.assertIn("[M] Four-Group Unified Simulation Comparison", combined_output)
        self.assertIn("[N] Virtual-Real Two-Stage Workflow", combined_output)

    def test_start_en_bat_menu_shows_runtime_log_mode_toggle(self):
        _, combined_output = self._run_start_en_bat()
        self.assertIn("Current Runtime Log Mode: User Mode", combined_output)
        self.assertIn("[T] Toggle Runtime Log Mode (Current Session)", combined_output)

    def test_start_en_bat_can_toggle_runtime_log_mode_for_current_session(self):
        _, combined_output = self._run_start_en_bat("T\r\n")
        self.assertIn("Switched to Detail Mode.", combined_output)

    def test_start_en_bat_menu_reflects_detail_runtime_log_mode(self):
        _, combined_output = self._run_start_en_bat(extra_env={"AIRSIM_RUNTIME_LOG_MODE": "detail"})
        self.assertIn("Current Runtime Log Mode: Detail Mode", combined_output)

    def test_start_en_bat_comparison_workflow_offers_resume_action_for_unfinished_run(self):
        temp_root = Path(__file__).resolve().parent / ".codex_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        workspace_root = temp_root / f"start_en_bat_{uuid.uuid4().hex}"
        exp_root = workspace_root / "analysis_results" / "workflows" / "comparison" / "2026-04-16_101732_comparison"
        exp_root.mkdir(parents=True, exist_ok=True)
        (exp_root / "workflow_state.json").write_text(
            json.dumps(
                {
                    "workflow_type": "comparison",
                    "experiment_id": exp_root.name,
                    "status": "running",
                    "current_phase": "stage01_ddpg",
                    "steps": {"stage01_ddpg": {"status": "running"}},
                    "updated_at": "2026-04-16 10:17:32",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        capture_file = workspace_root / "workflow_call.txt"
        _, combined_output = self._run_start_en_bat(
            "M\r\n",
            extra_env={
                "AIRSIM_TEST_EXIT_AFTER_WORKFLOW": "1",
                "AIRSIM_TEST_SKIP_QUICK_CONFIG": "1",
                "AIRSIM_TEST_WORKFLOW_ACTION": "C",
                "AIRSIM_WORKFLOW_WORKSPACE_ROOT": str(workspace_root),
                "AIRSIM_TEST_PAPER_WORKFLOW_CAPTURE_FILE": str(capture_file),
            },
        )

        self.assertIn("Unfinished workflow detected", combined_output)
        self.assertIn("Resume the latest workflow", combined_output)
        self.assertNotIn("Invalid selection. Please try again.", combined_output)
        self.assertEqual(capture_file.read_text(encoding="utf-8").strip(), "comparison|resume")
        shutil.rmtree(workspace_root, ignore_errors=True)
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
