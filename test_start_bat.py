import os
import subprocess
import unittest
from pathlib import Path


@unittest.skipUnless(os.name == "nt", "start.bat is Windows-only")
class StartBatTests(unittest.TestCase):
    def _run_start_bat(self):
        script = Path(__file__).resolve().parent / "start.bat"
        completed = subprocess.run(
            ["cmd.exe", "/d", "/c", str(script)],
            input="0\r\n",
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(script.parent),
        )
        combined_output = f"{completed.stdout}\n{completed.stderr}"
        return completed, combined_output

    def test_start_bat_menu_exits_without_cmd_parse_errors(self):
        completed, combined_output = self._run_start_bat()
        self.assertEqual(completed.returncode, 0, msg=combined_output)
        self.assertIn("AirSim", combined_output)
        self.assertNotIn(
            "not recognized as an internal or external command", combined_output
        )

    def test_start_bat_menu_mentions_virtual_real_two_stage_workflow(self):
        _, combined_output = self._run_start_bat()
        self.assertIn("Virtual-Real Two-Stage Workflow", combined_output)


if __name__ == "__main__":
    unittest.main()
