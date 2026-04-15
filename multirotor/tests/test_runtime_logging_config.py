import io
import logging
import os
import shutil
import unittest
from pathlib import Path

from multirotor.Algorithm._test_temp_paths import make_temp_dir
from multirotor.runtime_logging import configure_runtime_logging, should_emit_detail_feedback


class RuntimeLoggingConfigTests(unittest.TestCase):
    def setUp(self):
        self.log_dir = make_temp_dir("runtime_logging")
        self._prev_mode = os.environ.get("AIRSIM_RUNTIME_LOG_MODE")
        self._prev_dir = os.environ.get("AIRSIM_RUNTIME_LOG_DIR")
        os.environ["AIRSIM_RUNTIME_LOG_MODE"] = "user"
        os.environ["AIRSIM_RUNTIME_LOG_DIR"] = str(self.log_dir)

    def tearDown(self):
        if self._prev_mode is None:
            os.environ.pop("AIRSIM_RUNTIME_LOG_MODE", None)
        else:
            os.environ["AIRSIM_RUNTIME_LOG_MODE"] = self._prev_mode
        if self._prev_dir is None:
            os.environ.pop("AIRSIM_RUNTIME_LOG_DIR", None)
        else:
            os.environ["AIRSIM_RUNTIME_LOG_DIR"] = self._prev_dir
        shutil.rmtree(self.log_dir, ignore_errors=True)

    def test_user_mode_console_filter_keeps_connection_status_but_suppresses_high_frequency_noise(self):
        stream = io.StringIO()
        result = configure_runtime_logging(force_reconfigure=True, console_stream=stream)

        algorithm_logger = logging.getLogger("AlgorithmServer")
        unity_logger = logging.getLogger("UnitySocketServer")
        collector_logger = logging.getLogger("DataCollector")
        drone_logger = logging.getLogger("DroneController")

        algorithm_logger.info("等待 Unity 连接中... 1.0 秒")
        unity_logger.info("已发送无人机配置到Unity")
        collector_logger.info("结束 Episode 3 并保存（奖励: 12.34, 步数: 56）")
        algorithm_logger.warning("🔴 [网格更新] 收到246个格子，平均熵值=79.8, 低熵格子=3")
        drone_logger.warning("无人机UAV1发生碰撞: 对象=Room_diban_01, 穿透深度=0.011m")
        collector_logger.warning("训练数据刷盘失败: disk full")

        output = stream.getvalue()
        self.assertIn("等待 Unity 连接中... 1.0 秒", output)
        self.assertIn("训练数据刷盘失败: disk full", output)
        self.assertNotIn("已发送无人机配置到Unity", output)
        self.assertNotIn("结束 Episode 3 并保存", output)
        self.assertNotIn("🔴 [网格更新]", output)
        self.assertNotIn("发生碰撞", output)
        self.assertTrue(Path(result["log_file"]).exists())
        file_output = Path(result["log_file"]).read_text(encoding="utf-8")
        self.assertIn("🔴 [网格更新]", file_output)
        self.assertIn("发生碰撞", file_output)

    def test_user_mode_disables_reward_detail_feedback(self):
        self.assertFalse(should_emit_detail_feedback())

    def test_detail_mode_enables_reward_detail_feedback(self):
        os.environ["AIRSIM_RUNTIME_LOG_MODE"] = "detail"
        self.assertTrue(should_emit_detail_feedback())


if __name__ == "__main__":
    unittest.main()
