import os
import subprocess
import unittest
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


if __name__ == "__main__":
    unittest.main()
