import ast
import unittest
from pathlib import Path


def _iter_static_string_parts(node: ast.AST):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield node.value
        return
    if isinstance(node, ast.JoinedStr):
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                yield value.value


def _is_logger_or_print_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name) and func.id == "print":
        return True
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        if func.value.id in {"logger", "logging"}:
            return True
    return False


class RuntimeLoggingTextTests(unittest.TestCase):
    def test_runtime_logging_calls_do_not_contain_corrupted_question_mark_placeholders(self):
        root = Path(__file__).resolve().parents[2]
        targets = [
            root / "multirotor" / "AlgorithmServer.py",
            root / "multirotor" / "AirsimServer" / "drone_controller.py",
            root / "multirotor" / "Algorithm" / "data_collector.py",
        ]

        offenders = []
        for path in targets:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not _is_logger_or_print_call(node):
                    continue
                for arg in node.args:
                    for text in _iter_static_string_parts(arg):
                        if "??" in text:
                            offenders.append(f"{path.relative_to(root)}:{node.lineno}: {text}")

        self.assertEqual(
            offenders,
            [],
            msg="Corrupted runtime log strings still contain '??' placeholders:\n" + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
