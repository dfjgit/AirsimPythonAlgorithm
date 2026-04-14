import os
import subprocess
import unittest
from pathlib import Path


@unittest.skipUnless(os.name == "nt", "start.bat is Windows-only")
class StartBatTests(unittest.TestCase):
    def test_start_bat_menu_exits_without_cmd_parse_errors(self):
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

        self.assertEqual(completed.returncode, 0, msg=combined_output)
        self.assertIn("AirSim", combined_output)
        self.assertIn("[M] 论文对比分析实验工作流", combined_output)
        self.assertNotIn("not recognized as an internal or external command", combined_output)


if __name__ == "__main__":
    unittest.main()
