"""Tests for HAL health dashboard."""
import os
import unittest

from lib.hal.health import HealthCheck, HealthDashboard, HealthReport, _dir_size
from lib.hal.home import HAL_HOME as _HAL_HOME


class TestHealthCheck(unittest.TestCase):

    def test_ok_check(self):
        check = HealthCheck(name="test", status="ok", detail="all good")
        self.assertEqual(check.status, "ok")

    def test_frozen(self):
        check = HealthCheck(name="test", status="ok", detail="x")
        with self.assertRaises(AttributeError):
            check.name = "changed"


class TestHealthReport(unittest.TestCase):

    def test_overall_ok(self):
        report = HealthReport(checks=[
            HealthCheck("a", "ok", "fine"),
            HealthCheck("b", "ok", "fine"),
        ])
        self.assertEqual(report.overall, "ok")

    def test_overall_warn(self):
        report = HealthReport(checks=[
            HealthCheck("a", "ok", "fine"),
            HealthCheck("b", "warn", "iffy"),
        ])
        self.assertEqual(report.overall, "warn")

    def test_overall_error(self):
        report = HealthReport(checks=[
            HealthCheck("a", "ok", "fine"),
            HealthCheck("b", "error", "broken"),
        ])
        self.assertEqual(report.overall, "error")

    def test_summary(self):
        report = HealthReport(checks=[
            HealthCheck("a", "ok", "x"),
            HealthCheck("b", "warn", "y"),
        ])
        self.assertIn("warn", report.summary)
        self.assertIn("1 ok", report.summary)

    def test_to_dict(self):
        report = HealthReport(checks=[HealthCheck("a", "ok", "x")])
        d = report.to_dict()
        self.assertEqual(d["overall"], "ok")
        self.assertEqual(len(d["checks"]), 1)


class TestHealthDashboard(unittest.TestCase):

    def test_runs_without_registry(self):
        dashboard = HealthDashboard()
        report = dashboard.check()
        self.assertGreater(len(report.checks), 0)

    def test_hal_home_check(self):
        dashboard = HealthDashboard()
        check = dashboard._check_hal_home()
        self.assertEqual(check.status, "ok")

    def test_modules_check(self):
        dashboard = HealthDashboard()
        check = dashboard._check_modules()
        self.assertEqual(check.status, "ok")
        self.assertIn("modules", check.detail)

    def test_check_default_reveals_path(self):
        # Direct/CLI use (this module's own docstring) is unchanged: calling
        # check() with no argument still returns the literal HAL_HOME path.
        dashboard = HealthDashboard()
        hal_home = next(c for c in dashboard.check().checks if c.name == "hal_home")
        self.assertEqual(hal_home.detail, _HAL_HOME)

    def test_check_reveal_path_false_redacts_detail(self):
        # The public /api/health route passes reveal_path=False for a
        # non-loopback caller — the absolute path must not appear anywhere
        # in the redacted detail.
        dashboard = HealthDashboard()
        hal_home = next(
            c for c in dashboard.check(reveal_path=False).checks if c.name == "hal_home"
        )
        self.assertNotIn(_HAL_HOME, hal_home.detail)
        self.assertEqual(hal_home.detail, "configured")

    def test_check_reveal_path_false_preserves_name_and_status(self):
        # A monitor keyed on name/status (never on detail) must behave
        # identically regardless of reveal_path.
        dashboard = HealthDashboard()
        full = dashboard.check(reveal_path=True)
        redacted = dashboard.check(reveal_path=False)
        full_shape = [(c.name, c.status) for c in full.checks]
        redacted_shape = [(c.name, c.status) for c in redacted.checks]
        self.assertEqual(full_shape, redacted_shape)
        self.assertEqual(full.overall, redacted.overall)

    def test_hal_home_check_reveal_path_false(self):
        dashboard = HealthDashboard()
        check = dashboard._check_hal_home(reveal_path=False)
        self.assertEqual(check.status, "ok")
        self.assertEqual(check.detail, "configured")


class TestDirSize(unittest.TestCase):

    def test_empty_dir(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(_dir_size(tmpdir), 0)

    def test_with_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.txt")
            with open(path, "w") as f:
                f.write("hello world")
            size = _dir_size(tmpdir)
            self.assertGreater(size, 0)


if __name__ == "__main__":
    unittest.main()
