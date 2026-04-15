import unittest
import json
from pathlib import Path


UTF8_BOM = b"\xef\xbb\xbf"


class BatchEntryScriptTests(unittest.TestCase):
    def test_start_launchers_define_ui_language(self):
        root = Path(__file__).resolve().parent
        self.assertIn('set "AIRSIM_UI_LANG=zh"', (root / "start.bat").read_text(encoding="utf-8"))
        self.assertIn('set "AIRSIM_UI_LANG=en"', (root / "start_en.bat").read_text(encoding="utf-8"))

    def test_start_launchers_default_to_user_runtime_log_mode(self):
        root = Path(__file__).resolve().parent
        self.assertIn('set "AIRSIM_RUNTIME_LOG_MODE=user"', (root / "start.bat").read_text(encoding="utf-8"))
        self.assertIn('set "AIRSIM_RUNTIME_LOG_MODE=user"', (root / "start_en.bat").read_text(encoding="utf-8"))

    def test_start_launchers_expose_runtime_log_mode_toggle(self):
        root = Path(__file__).resolve().parent
        start_bat = (root / "start.bat").read_text(encoding="utf-8")
        start_en_bat = (root / "start_en.bat").read_text(encoding="utf-8")
        self.assertIn("[T] 切换运行时日志模式（当前会话）", start_bat)
        self.assertIn(":toggle_runtime_log_mode", start_bat)
        self.assertIn("[T] Toggle Runtime Log Mode (Current Session)", start_en_bat)
        self.assertIn(":toggle_runtime_log_mode", start_en_bat)

    def test_start_bat_comparison_workflow_preview_mentions_all_four_algorithm_groups(self):
        root = Path(__file__).resolve().parent
        start_bat = (root / "start.bat").read_text(encoding="utf-8")
        self.assertIn("本流程将依次执行以下阶段：", start_bat)
        self.assertIn("APF 基线多轮仿真阶段", start_bat)
        self.assertIn("  [1] APF 基线多轮仿真阶段（fixed APF / random APF）", start_bat)
        self.assertIn("  [2] DDPG+APF stage01 训练", start_bat)
        self.assertIn("  [3] Pure DQN stage01 训练", start_bat)
        self.assertIn("在 Unity/AirSim 中执行四组仿真评测（冻结策略：fixed APF / random APF / DDPG+APF / Pure DQN）", start_bat)
        self.assertIn("fixed APF 与 random APF 不参加训练阶段，但会先进行多轮仿真，再进入最终统一对比", start_bat)
        self.assertIn("请输入 Y 继续执行，输入其它任意键返回主菜单：", start_bat)

    def test_start_bat_mentions_separate_quick_config_fields_for_apf_and_benchmark_episodes(self):
        root = Path(__file__).resolve().parent
        start_bat = (root / "start.bat").read_text(encoding="utf-8")
        self.assertIn("无人机数量", start_bat)
        self.assertIn("APF 基线多轮仿真轮次", start_bat)
        self.assertIn("四组 benchmark 每 seed 评测轮次", start_bat)

    def test_workflow_profiles_include_drone_count_quick_config(self):
        root = Path(__file__).resolve().parent
        schema = json.loads((root / "scripts" / "start_quick_config_schema.json").read_text(encoding="utf-8"))
        self.assertEqual(schema["profiles"]["comparison_workflow"]["fields"][0], "drone_count")
        self.assertEqual(schema["profiles"]["two_stage_workflow"]["fields"][0], "drone_count")

    def test_visualization_quick_config_is_available_for_simulation_and_training_profiles(self):
        root = Path(__file__).resolve().parent
        schema = json.loads((root / "scripts" / "start_quick_config_schema.json").read_text(encoding="utf-8"))
        self.assertIn("simulation_visualization", schema["fields"])
        self.assertEqual(schema["fields"]["simulation_visualization"]["env"], "AIRSIM_QUICK_VISUALIZATION")
        for profile_name in [
            "ddpg_train",
            "dqn_train",
            "dqn_resume_train",
            "comparison_workflow",
            "four_group_benchmark",
            "paper_ddpg_seeds",
            "paper_dqn_seeds",
            "two_stage_workflow",
        ]:
            self.assertIn(
                "simulation_visualization",
                schema["profiles"][profile_name]["fields"],
                msg=profile_name,
            )

    def test_workflow_launchers_support_dot_venv_python(self):
        root = Path(__file__).resolve().parent
        workflow_bat = (root / "scripts" / "Run_Paper_Workflow.bat").read_text(encoding="utf-8")
        apf_baseline_bat = (root / "scripts" / "Run_APF_Baseline_Simulation.bat").read_text(encoding="utf-8")
        self.assertIn(".venv\\Scripts\\python.exe", workflow_bat)
        self.assertIn(".venv\\Scripts\\python.exe", apf_baseline_bat)

    def test_start_bat_workflow_entries_report_failure_before_returning_to_menu(self):
        root = Path(__file__).resolve().parent
        start_bat = (root / "start.bat").read_text(encoding="utf-8")
        self.assertIn("四组统一仿真对比阶段执行失败", start_bat)
        self.assertIn("虚实两阶段实验工作流执行失败", start_bat)

    def test_quick_config_document_is_linked_from_user_visible_surfaces(self):
        root = Path(__file__).resolve().parent
        self.assertTrue((root / "docs" / "START_QUICK_CONFIG_ZH.md").exists())
        self.assertIn("docs\\START_QUICK_CONFIG_ZH.md", (root / "start.bat").read_text(encoding="utf-8"))
        self.assertIn("START_QUICK_CONFIG_ZH.md", (root / "README.md").read_text(encoding="utf-8"))

    def test_start_entry_scripts_honor_ui_language_context(self):
        root = Path(__file__).resolve().parent
        scripts_dir = root / "scripts"
        localized_scripts = [
            "Run_System_Fixed_Weights.bat",
            "Run_System_DDPG_Weights.bat",
            "Train_DDPG_Weights_Real_Environment.bat",
            "Train_DDPG_Weights_Crazyflie_Logs.bat",
            "Train_DQN_Movement_Real_Environment.bat",
            "Train_Hierarchical_DQN.bat",
            "Train_Hierarchical_With_AirSim.bat",
            "Test_DQN_Movement.bat",
            "Data_Visualization_Analysis.bat",
            "Run_APF_Baseline_Simulation.bat",
            "Run_Four_Group_Benchmark.bat",
            "Analyze_Four_Group_Benchmark.bat",
            "Analyze_Family_Comparisons.bat",
        ]

        offenders = []
        for name in localized_scripts:
            content = (scripts_dir / name).read_text(encoding="utf-8")
            if "AIRSIM_UI_LANG" not in content:
                offenders.append(name)

        self.assertEqual(
            offenders,
            [],
            msg="These Start-reachable scripts do not honor AIRSIM_UI_LANG: " + ", ".join(offenders),
        )

    def test_simulation_and_training_entrypoints_honor_quick_visualization_override(self):
        root = Path(__file__).resolve().parent
        expected_files = [
            root / "multirotor" / "Algorithm" / "four_group_benchmark_runner.py",
            root / "multirotor" / "DDPG_Weight" / "train_with_airsim_improved.py",
            root / "multirotor" / "DQN_Movement" / "scripts" / "train_movement_with_airsim.py",
        ]

        for path in expected_files:
            content = path.read_text(encoding="utf-8-sig")
            self.assertIn("AIRSIM_QUICK_VISUALIZATION", content, msg=str(path))

    def test_chinese_visualization_menu_uses_chinese_comparison_title(self):
        root = Path(__file__).resolve().parent
        content = (root / "scripts" / "数据可视化分析.bat").read_text(encoding="utf-8")
        self.assertIn("统一多算法对比分析", content)
        self.assertNotIn("Unified Multi-Algorithm Comparison", content)

    def test_chinese_single_episode_launcher_uses_chinese_descriptive_copy(self):
        root = Path(__file__).resolve().parent
        content = (root / "scripts" / "Train_DDPG_Weights_Crazyflie_Online_Single_Episode.bat").read_text(encoding="utf-8")
        self.assertIn("实体无人机在线飞行 1 个训练回合", content)
        self.assertIn("在回合结束后执行高权重更新", content)

    def test_core_chinese_entry_scripts_use_polished_copy(self):
        root = Path(__file__).resolve().parent / "scripts"
        expected_phrases = {
            "运行系统-固定权重.bat": "切换到运行目录并启动算法服务器...",
            "运行系统-DDPG权重.bat": "请先运行选项 [4] 启动 DDPG+APF 训练",
            "测试移动DQN.bat": "本脚本将验证已训练的 DQN 控制模型",
            "训练权重DDPG-真实环境.bat": "本脚本将在 Unity AirSim 仿真环境中训练 DDPG 权重模型",
            "训练权重DDPG-实体机日志.bat": "日志仅用于离线训练，不会控制实体无人机",
            "训练移动DQN-真实环境.bat": "本脚本将在 Unity AirSim 仿真环境中训练 DQN 控制模型",
        }

        for file_name, expected_phrase in expected_phrases.items():
            content = (root / file_name).read_text(encoding="utf-8")
            self.assertIn(expected_phrase, content, msg=file_name)

    def test_four_group_benchmark_launcher_explicitly_lists_all_simulated_groups_in_chinese(self):
        root = Path(__file__).resolve().parent
        content = (root / "scripts" / "Run_Four_Group_Benchmark.bat").read_text(encoding="utf-8")
        self.assertIn("本阶段将依次在 Unity/AirSim 中评测以下四组：", content)
        self.assertIn("fixed APF（固定策略基线，不参加训练）", content)
        self.assertIn("random APF（随机策略基线，不参加训练）", content)
        self.assertIn("DDPG+APF（使用已训练模型，冻结策略）", content)
        self.assertIn("Pure DQN（使用已训练模型，冻结策略）", content)

    def test_start_reachable_batch_scripts_use_plain_utf8_without_bom(self):
        root = Path(__file__).resolve().parent
        batch_files = [root / "start.bat", root / "start_en.bat", *sorted((root / "scripts").glob("*.bat"))]

        offenders = []
        for path in batch_files:
            data = path.read_bytes()
            if data.startswith(UTF8_BOM):
                offenders.append(str(path.relative_to(root)))

        self.assertEqual(
            offenders,
            [],
            msg=(
                "These batch entry scripts still use UTF-8 BOM, which can break "
                "@echo off and cause command-by-command echo in cmd.exe: "
                + ", ".join(offenders)
            ),
        )

    def test_start_reachable_batch_scripts_begin_with_echo_off(self):
        root = Path(__file__).resolve().parent
        batch_files = [root / "start.bat", root / "start_en.bat", *sorted((root / "scripts").glob("*.bat"))]

        offenders = []
        for path in batch_files:
            first_line = path.read_text(encoding="utf-8").splitlines()[0].strip()
            if first_line != "@echo off":
                offenders.append(f"{path.relative_to(root)} -> {first_line!r}")

        self.assertEqual(
            offenders,
            [],
            msg=(
                "These batch entry scripts do not start with '@echo off': "
                + ", ".join(offenders)
            ),
        )

    def test_start_reachable_batch_scripts_use_consistent_windows_crlf_line_endings(self):
        root = Path(__file__).resolve().parent
        batch_files = [root / "start.bat", root / "start_en.bat", *sorted((root / "scripts").glob("*.bat"))]

        offenders = []
        for path in batch_files:
            data = path.read_bytes()
            lf_count = data.count(b"\n")
            crlf_count = data.count(b"\r\n")
            if lf_count != crlf_count:
                offenders.append(str(path.relative_to(root)))

        self.assertEqual(
            offenders,
            [],
            msg=(
                "These batch entry scripts mix LF and CRLF line endings, which can break cmd.exe parsing for Chinese menu lines: "
                + ", ".join(offenders)
            ),
        )


if __name__ == "__main__":
    unittest.main()
