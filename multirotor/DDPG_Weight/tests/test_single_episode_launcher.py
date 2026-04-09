import os
import shutil
import time
import unittest
from pathlib import Path

from multirotor.DDPG_Weight.single_episode_launcher import (
    normalize_model_path_input,
    resolve_default_model,
)


class SingleEpisodeLauncherTests(unittest.TestCase):
    def _tempdir(self) -> Path:
        safe_name = self.id().rsplit(".", 1)[-1]
        temp_root = Path(__file__).resolve().parent / "_tmp_single_episode_launcher" / safe_name
        if temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)
        temp_root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(temp_root, ignore_errors=True))
        return temp_root

    def test_resolve_default_model_prefers_latest_online_model(self):
        models_dir = self._tempdir()
        older = models_dir / "weight_predictor_crazyflie_online_20260408_090000.zip"
        newer = models_dir / "weight_predictor_crazyflie_online_20260409_090000.zip"
        older.touch()
        newer.touch()
        os.utime(older, (time.time() - 20, time.time() - 20))
        os.utime(newer, (time.time(), time.time()))

        result = resolve_default_model(models_dir)

        self.assertEqual(result.status, "online")
        self.assertEqual(result.model_path, str(newer.with_suffix("")))

    def test_resolve_default_model_falls_back_to_airsim_model(self):
        models_dir = self._tempdir()
        airsim_model = models_dir / "weight_predictor_airsim.zip"
        airsim_model.touch()

        result = resolve_default_model(models_dir)

        self.assertEqual(result.status, "airsim")
        self.assertEqual(result.model_path, str(airsim_model.with_suffix("")))

    def test_resolve_default_model_returns_missing_when_no_model_exists(self):
        result = resolve_default_model(self._tempdir())

        self.assertEqual(result.status, "missing")
        self.assertIsNone(result.model_path)

    def test_normalize_model_path_input_strips_quotes_and_zip_suffix(self):
        raw_path = '"D:\\Models\\weight_predictor_crazyflie_online_20260409_090000.zip"'

        normalized = normalize_model_path_input(raw_path)

        self.assertEqual(
            normalized,
            "D:\\Models\\weight_predictor_crazyflie_online_20260409_090000",
        )


if __name__ == "__main__":
    unittest.main()
