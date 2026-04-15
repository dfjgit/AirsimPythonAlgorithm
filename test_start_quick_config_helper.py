import subprocess
import shutil
import unittest
from pathlib import Path

from multirotor.Algorithm._test_temp_paths import make_temp_dir


class StartQuickConfigHelperTests(unittest.TestCase):
    def test_helper_writes_defaults_for_comparison_workflow_on_blank_input(self):
        root = Path(__file__).resolve().parent
        helper = root / "scripts" / "start_quick_config_helper.py"
        schema = root / "scripts" / "start_quick_config_schema.json"
        temp_dir = make_temp_dir("start_quick_config_helper")
        try:
            output_file = temp_dir / "quick.env"
            completed = subprocess.run(
                ["python", str(helper), "--schema", str(schema), "--profile", "comparison_workflow", "--output", str(output_file), "--lang", "zh"],
                input="\n\n\n\n\n\n",
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(root),
            )

            self.assertEqual(completed.returncode, 0, msg=completed.stderr)
            content = output_file.read_text(encoding="utf-8")
            self.assertIn("AIRSIM_QUICK_DRONES=", content)
            self.assertIn("AIRSIM_QUICK_APF_BASELINE_EPISODES=", content)
            self.assertIn("AIRSIM_QUICK_BENCHMARK_EPISODES=", content)
            self.assertIn("AIRSIM_QUICK_DDPG_TIMESTEPS=", content)
            self.assertIn("AIRSIM_QUICK_DQN_TIMESTEPS=", content)
            self.assertIn("AIRSIM_QUICK_VISUALIZATION=", content)
            self.assertIn("四组统一仿真对比阶段 - 快速配置", completed.stderr)
            self.assertIn("本次执行配置摘要", completed.stderr)
            self.assertIn("APF 基线多轮仿真轮次", completed.stderr)
            self.assertIn("四组 benchmark 每 seed 评测轮次", completed.stderr)
            self.assertIn("无人机数量", completed.stderr)
            self.assertIn("仿真可视化窗口", completed.stderr)
            self.assertIn("步数对应的是仿真时间预估，不等同于真实墙钟耗时", completed.stderr)
            self.assertIn("用于 fixed APF / random APF 的基线多轮仿真阶段", completed.stderr)
            self.assertIn("用于四组最终统一仿真对比阶段", completed.stderr)
            self.assertIn("2.0 秒/步", completed.stderr)
            self.assertIn("1.5 秒/步", completed.stderr)
            self.assertIn("约 12.0 小时仿真时间", completed.stderr)
            self.assertIn("约 12.5 小时仿真时间", completed.stderr)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_helper_emits_english_copy_for_comparison_workflow(self):
        root = Path(__file__).resolve().parent
        helper = root / "scripts" / "start_quick_config_helper.py"
        schema = root / "scripts" / "start_quick_config_schema.json"
        temp_dir = make_temp_dir("start_quick_config_helper_en")
        try:
            output_file = temp_dir / "quick.env"
            completed = subprocess.run(
                ["python", str(helper), "--schema", str(schema), "--profile", "comparison_workflow", "--output", str(output_file), "--lang", "en"],
                input="\n\n\n\n\n\n",
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(root),
            )

            self.assertEqual(completed.returncode, 0, msg=completed.stderr)
            self.assertIn("Four-Group Unified Simulation Comparison - Quick Config", completed.stderr)
            self.assertIn("Drone Count", completed.stderr)
            self.assertIn("APF Baseline Simulation Episodes", completed.stderr)
            self.assertIn("DDPG Training Steps", completed.stderr)
            self.assertIn("Simulation Visualization Window", completed.stderr)
            self.assertIn("2.0 sec/step", completed.stderr)
            self.assertIn("Execution Summary", completed.stderr)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
