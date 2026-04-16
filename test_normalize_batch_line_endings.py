import subprocess
import unittest
import uuid
from pathlib import Path


class NormalizeBatchLineEndingsTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parent
        self.script = self.repo_root / "scripts" / "normalize_batch_line_endings.py"
        self.temp_root = self.repo_root / "tmp" / f"normalize_batch_eol_{uuid.uuid4().hex}"
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if not self.temp_root.exists():
            return
        for child in sorted(self.temp_root.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        self.temp_root.rmdir()

    def test_check_reports_mixed_line_endings(self):
        target = self.temp_root / "broken.bat"
        target.write_bytes(b"@echo off\r\necho ok\n")

        completed = subprocess.run(
            ["python", str(self.script), "--check", str(target)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(self.repo_root),
            timeout=8,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("needs CRLF normalization", completed.stdout)

    def test_write_normalizes_batch_file_to_crlf(self):
        target = self.temp_root / "broken.bat"
        target.write_bytes(b"@echo off\r\necho ok\n")

        completed = subprocess.run(
            ["python", str(self.script), "--write", str(target)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(self.repo_root),
            timeout=8,
        )

        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        normalized = target.read_bytes()
        self.assertEqual(normalized, b"@echo off\r\necho ok\r\n")


if __name__ == "__main__":
    unittest.main()
