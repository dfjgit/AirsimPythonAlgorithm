import os
import subprocess
import unittest
from pathlib import Path


@unittest.skipUnless(os.name == "nt", "start.bat is Windows-only")
class StartBatTests(unittest.TestCase):
    def _run_start_bat(self, input_text="0\r\n", extra_env=None):
        script = Path(__file__).resolve().parent / "start.bat"
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

    def test_start_bat_menu_exits_without_cmd_parse_errors(self):
        completed, combined_output = self._run_start_bat()
        self.assertEqual(completed.returncode, 0, msg=combined_output)
        self.assertIn("AirSim", combined_output)
        self.assertIn("控制台", combined_output)
        self.assertNotIn(
            "not recognized as an internal or external command", combined_output
        )

    def test_start_bat_menu_mentions_virtual_real_two_stage_workflow(self):
        _, combined_output = self._run_start_bat()
        self.assertIn("Virtual-Real Two-Stage Workflow", combined_output)

    def test_start_bat_menu_mentions_four_group_paper_experiment_entry(self):
        _, combined_output = self._run_start_bat()
        self.assertIn("[M] 四组统一仿真对比阶段", combined_output)

    def test_start_bat_menu_uses_formal_product_copy(self):
        _, combined_output = self._run_start_bat()
        self.assertIn("=== 结果分析 ===", combined_output)
        self.assertIn("[C] 清理训练与分析产出", combined_output)

    def test_start_bat_menu_shows_runtime_log_mode_toggle(self):
        _, combined_output = self._run_start_bat()
        self.assertIn("当前运行时日志模式: 用户模式", combined_output)
        self.assertIn("[T] 切换运行时日志模式（当前会话）", combined_output)

    def test_start_bat_can_toggle_runtime_log_mode_for_current_session(self):
        _, combined_output = self._run_start_bat("T\r\n")
        self.assertIn("已切换到详细模式。", combined_output)

    def test_start_bat_menu_reflects_detail_runtime_log_mode(self):
        _, combined_output = self._run_start_bat(extra_env={"AIRSIM_RUNTIME_LOG_MODE": "detail"})
        self.assertIn("当前运行时日志模式: 详细模式", combined_output)


if __name__ == "__main__":
    unittest.main()
