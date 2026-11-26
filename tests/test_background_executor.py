import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import tempfile
import json
import time

from lib.hal.background_executor import (
    BackgroundTask,
    BackgroundExecutor,
    TaskRunner,
    TaskStatus,
)


def _make_task(**kwargs) -> BackgroundTask:
    defaults = dict(
        task_id="abc12345",
        agent_id="agent-1",
        session_id="bg-abc12345",
        description="Test task",
        created_at=time.time(),
    )
    defaults.update(kwargs)
    return BackgroundTask(**defaults)


def _mock_response() -> MagicMock:
    resp = MagicMock()
    resp.text = "test"
    resp.model = "test/model"
    resp.provider = "test"
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
    resp.cost_usd = 0.001
    return resp


class TestBackgroundExecutor(unittest.TestCase):

    def test_task_to_dict_keys(self):
        task = _make_task()
        result = task.to_dict()
        self.assertEqual(len(result), 12)
        expected = {
            "task_id", "agent_id", "session_id", "description", "status",
            "created_at", "started_at", "completed_at", "result", "error",
            "cost_usd", "turns_completed",
        }
        self.assertEqual(set(result.keys()), expected)

    def test_task_summary_keys(self):
        task = _make_task()
        result = task.summary()
        self.assertEqual(
            set(result.keys()),
            {"task_id", "agent_id", "description", "status", "turns_completed", "cost_usd"},
        )

    def test_task_store_roundtrip(self):
        from lib.hal.background_executor import TaskStore
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("lib.hal.background_executor._TASKS_DIR", tmpdir):
                store = TaskStore()
                task = _make_task(task_id="roundtrip")
                store.save(task)
                loaded = store.load("roundtrip")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.task_id, "roundtrip")
        self.assertEqual(loaded.description, "Test task")

    def test_task_runner_fail_agent_not_found(self):
        import threading
        task = _make_task()
        agents = MagicMock()
        agents.get.return_value = None
        emitter = MagicMock()
        runner = TaskRunner(task, agents, emitter, threading.Event())
        runner.run()
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertIn("not found", task.error)

    def test_executor_submit_returns_8_char_id(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            with patch("lib.hal.background_executor._TASKS_DIR", tmpdir):
                agents = MagicMock()
                agents.get.return_value = MagicMock(
                    model="test/model", system_prompt="You are helpful."
                )
                with patch("lib.hal.registry.get_registry") as mock_reg:
                    mock_reg.return_value = MagicMock()
                    mock_reg.return_value.chat.return_value = _mock_response()
                    mock_reg.return_value.routing.default_model = "test/model"
                    executor = BackgroundExecutor(agents)
                    task_id = executor.submit("agent-1", "Do something")
                    import time; time.sleep(0.1)  # let daemon thread finish
        self.assertEqual(len(task_id), 8)

    def test_executor_cancel_queued(self):
        import threading

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("lib.hal.background_executor._TASKS_DIR", tmpdir):
                agents = MagicMock()
                agents.get.return_value = MagicMock(
                    model="test/model", system_prompt=""
                )

                executor = BackgroundExecutor(agents)

                # Patch thread start so the task stays QUEUED
                with patch.object(threading.Thread, "start", lambda self: None):
                    task_id = executor.submit("agent-1", "Do something slow")
                    result = executor.cancel(task_id)

        self.assertTrue(result)

    def test_executor_list_tasks_sorted_most_recent_first(self):
        import threading
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            with patch("lib.hal.background_executor._TASKS_DIR", tmpdir), \
                 patch.object(threading.Thread, "start", lambda self: None):
                agents = MagicMock()
                agents.get.return_value = MagicMock(model="test/model", system_prompt="")

                executor = BackgroundExecutor(agents, store=MagicMock(load_all=lambda: []))

                id_older = executor.submit("agent-1", "Older task")
                executor._tasks[id_older].created_at = 1000.0
                id_newer = executor.submit("agent-1", "Newer task")
                executor._tasks[id_newer].created_at = 2000.0

                tasks = executor.list_tasks("agent-1")
                self.assertEqual(tasks[0].task_id, id_newer)
                self.assertEqual(tasks[1].task_id, id_older)


if __name__ == "__main__":
    unittest.main()
