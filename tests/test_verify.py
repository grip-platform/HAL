"""Tests for HAL post-edit verification."""
import os
import sys
import tempfile
import unittest

from lib.hal.verify import VerifyResult, Verifier


class TestVerifyResult(unittest.TestCase):

    def test_passed(self):
        r = VerifyResult(passed=True, tool="py_compile", errors="", file_path="test.py")
        self.assertTrue(r.passed)

    def test_failed(self):
        r = VerifyResult(passed=False, tool="tsc", errors="Type error", file_path="test.ts")
        self.assertFalse(r.passed)
        self.assertEqual(r.errors, "Type error")


class TestVerifierPython(unittest.TestCase):

    def test_valid_python(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("x = 1 + 2\nprint(x)\n")
            path = f.name
        try:
            v = Verifier()
            result = v.check(path)
            self.assertTrue(result.passed)
            self.assertEqual(result.tool, "py_compile")
        finally:
            os.remove(path)

    def test_invalid_python(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("def broken(\n")  # Syntax error
            path = f.name
        try:
            v = Verifier()
            result = v.check(path)
            self.assertFalse(result.passed)
            self.assertIn("py_compile", result.tool)
        finally:
            os.remove(path)

    def test_nonexistent_file(self):
        v = Verifier()
        result = v.check("/nonexistent/file.py")
        self.assertFalse(result.passed)
        self.assertEqual(result.tool, "exists")

    def test_unknown_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", mode="w", delete=False) as f:
            f.write("whatever")
            path = f.name
        try:
            v = Verifier()
            result = v.check(path)
            self.assertTrue(result.passed)  # No checker = pass
            self.assertEqual(result.tool, "none")
        finally:
            os.remove(path)


class TestVerifierCheckAll(unittest.TestCase):

    def test_multiple_files(self):
        files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
                f.write(f"x_{i} = {i}\n")
                files.append(f.name)
        try:
            v = Verifier()
            results = v.check_all(files)
            self.assertEqual(len(results), 3)
            self.assertTrue(all(r.passed for r in results))
        finally:
            for f in files:
                os.remove(f)


if __name__ == "__main__":
    unittest.main()
