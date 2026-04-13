import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from four_group_benchmark_analyzer import configure_plot_fonts


class FourGroupBenchmarkAnalyzerFontTests(unittest.TestCase):
    def test_configure_plot_fonts_prefers_chinese_capable_fonts(self):
        fonts = configure_plot_fonts()

        self.assertIn("Microsoft YaHei", fonts)
        self.assertIn("SimHei", fonts)


if __name__ == "__main__":
    unittest.main()
