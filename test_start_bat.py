import os
import shutil
import subprocess
import unittest
import uuid
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

    def test_start_bat_is_ascii_wrapper(self):
        script = Path(__file__).resolve().parent / "start.bat"
        data = script.read_bytes()
        self.assertTrue(
            all(byte in (9, 10, 13) or byte < 128 for byte in data),
            msg="start.bat should stay ASCII-only so cmd parsing does not depend on UTF-8 menu text",
        )

    def test_start_bat_wrapper_repairs_mixed_line_endings_before_launch(self):
        repo_root = Path(__file__).resolve().parent
        temp_root = repo_root / ".codex_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        workspace_root = temp_root / f"start_wrapper_{uuid.uuid4().hex}"
        (workspace_root / "scripts").mkdir(parents=True, exist_ok=True)

        shutil.copy2(repo_root / "start.bat", workspace_root / "start.bat")
        shutil.copy2(repo_root / "start_main.bat", workspace_root / "start_main.bat")
        shutil.copy2(
            repo_root / "scripts" / "Start_Batch_Bootstrap.ps1",
            workspace_root / "scripts" / "Start_Batch_Bootstrap.ps1",
        )

        main_file = workspace_root / "start_main.bat"
        lines = main_file.read_text(encoding="utf-8").splitlines()
        with main_file.open("wb") as handle:
            for idx, line in enumerate(lines, 1):
                handle.write(line.encode("utf-8"))
                handle.write(b"\n" if idx in {21, 24, 27, 30, 35, 40, 44, 47, 60, 61} else b"\r\n")

        env = dict(os.environ)
        env["AIRSIM_TEST_NO_PAUSE"] = "1"
        env["AIRSIM_TEST_EXIT_AFTER_TOGGLE"] = "1"
        completed = subprocess.run(
            ["cmd.exe", "/d", "/c", str(workspace_root / "start.bat")],
            input="0\r\n",
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(workspace_root),
            env=env,
            timeout=12,
        )
        combined_output = f"{completed.stdout}\n{completed.stderr}"

        self.assertEqual(completed.returncode, 0, msg=combined_output)
        self.assertIn("AirSim", combined_output)
        self.assertIn("控制台", combined_output)
        self.assertNotIn("not recognized as an internal or external command", combined_output)

        repaired = main_file.read_bytes()
        self.assertEqual(repaired.count(b"\n") - repaired.count(b"\r\n"), 0)

        shutil.rmtree(workspace_root, ignore_errors=True)
        shutil.rmtree(temp_root, ignore_errors=True)

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

    def test_batch_checkout_rules_are_versioned_for_crlf(self):
        repo_root = Path(__file__).resolve().parent
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", ".gitattributes"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(repo_root),
            timeout=8,
        )
        self.assertEqual(tracked.returncode, 0, msg=tracked.stderr)

        attrs = subprocess.run(
            ["git", "check-attr", "text", "--", "start.bat", "start_en.bat"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(repo_root),
            timeout=8,
        )
        self.assertEqual(attrs.returncode, 0, msg=attrs.stderr)
        self.assertIn("start.bat: text: unset", attrs.stdout)
        self.assertIn("start_en.bat: text: unset", attrs.stdout)

        for script_name in ("start.bat", "start_en.bat"):
            indexed = subprocess.run(
                ["git", "show", f":{script_name}"],
                capture_output=True,
                cwd=str(repo_root),
                timeout=8,
            )
            self.assertEqual(indexed.returncode, 0, msg=indexed.stderr.decode("utf-8", errors="replace"))
            blob = indexed.stdout
            self.assertIn(b"\r\n", blob, msg=f"{script_name} should keep CRLF in the indexed blob")
            self.assertEqual(
                blob.count(b"\n") - blob.count(b"\r\n"),
                0,
                msg=f"{script_name} should not keep bare LF in the indexed blob",
            )

    def test_start_bat_comparison_workflow_offers_resume_action_for_unfinished_run(self):
        temp_root = Path(__file__).resolve().parent / ".codex_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        workspace_root = temp_root / f"start_bat_{uuid.uuid4().hex}"
        exp_root = workspace_root / "analysis_results" / "workflows" / "comparison" / "2026-04-16_101732_comparison"
        exp_root.mkdir(parents=True, exist_ok=True)
        (exp_root / "workflow_state.json").write_text(
            '{\n'
            '  "workflow_type": "comparison",\n'
            f'  "experiment_id": "{exp_root.name}",\n'
            '  "status": "running",\n'
            '  "current_phase": "stage01_ddpg",\n'
            '  "steps": {\n'
            '    "stage01_ddpg": {\n'
            '      "status": "running"\n'
            '    }\n'
            '  },\n'
            '  "updated_at": "2026-04-16 10:17:32"\n'
            '}\n',
            encoding="utf-8",
        )
        capture_file = workspace_root / "workflow_call.txt"
        _, combined_output = self._run_start_bat(
            "M\r\n",
            extra_env={
                "AIRSIM_TEST_EXIT_AFTER_WORKFLOW": "1",
                "AIRSIM_TEST_SKIP_QUICK_CONFIG": "1",
                "AIRSIM_TEST_WORKFLOW_ACTION": "C",
                "AIRSIM_WORKFLOW_WORKSPACE_ROOT": str(workspace_root),
                "AIRSIM_TEST_PAPER_WORKFLOW_CAPTURE_FILE": str(capture_file),
            },
        )

        self.assertIn("检测到未完成的 workflow", combined_output)
        self.assertIn("继续当前实验", combined_output)
        self.assertNotIn("当前输入无效，请重新选择。", combined_output)
        self.assertNotIn("not recognized as an internal or external command", combined_output)
        self.assertEqual(capture_file.read_text(encoding="utf-8").strip(), "comparison|resume")
        shutil.rmtree(workspace_root, ignore_errors=True)
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
