"""Tests for KONO Bridge — HAL's semantic memory recall interface."""
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from lib.hal.kono_bridge import (
    _format_results,
    _kono_available,
    _MAX_CONTEXT_CHARS,
    clear_cache,
    recall_for_query,
)


class TestKonoBridgeRecall(unittest.TestCase):
    """Test the main recall_for_query entry point."""

    def setUp(self):
        clear_cache()

    def test_empty_query_returns_empty(self):
        self.assertEqual(recall_for_query(""), "")
        self.assertEqual(recall_for_query("   "), "")

    @patch("lib.hal.kono_bridge._kono_available", return_value=False)
    def test_kono_unavailable_returns_empty(self, _mock):
        result = recall_for_query("How does billing work?")
        self.assertEqual(result, "")

    @patch("lib.hal.kono_bridge._kono_available", return_value=True)
    @patch("lib.hal.kono_bridge._recall_with_timeout")
    def test_successful_recall(self, mock_recall, _mock_avail):
        mock_recall.return_value = "[KONO Memory]\n  1. Billing uses Stripe"
        result = recall_for_query("How does billing work?")
        self.assertIn("Billing uses Stripe", result)
        mock_recall.assert_called_once()

    @patch("lib.hal.kono_bridge._kono_available", return_value=True)
    @patch("lib.hal.kono_bridge._recall_with_timeout")
    def test_cache_prevents_duplicate_queries(self, mock_recall, _mock_avail):
        mock_recall.return_value = "cached result"
        recall_for_query("same question")
        recall_for_query("same question")
        # Should only call KONO once — second hit is cached
        mock_recall.assert_called_once()

    @patch("lib.hal.kono_bridge._kono_available", return_value=True)
    @patch("lib.hal.kono_bridge._recall_with_timeout")
    def test_cache_cleared_between_turns(self, mock_recall, _mock_avail):
        mock_recall.return_value = "result"
        recall_for_query("question")
        clear_cache()
        recall_for_query("question")
        self.assertEqual(mock_recall.call_count, 2)

    @patch("lib.hal.kono_bridge._kono_available", return_value=True)
    @patch("lib.hal.kono_bridge._recall_with_timeout", return_value="")
    def test_no_results_returns_empty(self, mock_recall, _mock_avail):
        result = recall_for_query("completely unknown topic xyz")
        self.assertEqual(result, "")


class TestKonoAvailability(unittest.TestCase):
    """Test KONO availability detection."""

    @patch("lib.hal.kono_bridge.DB_PATH", Path("/nonexistent/embeddings.db"))
    def test_missing_db(self):
        self.assertFalse(_kono_available())


class TestFormatResults(unittest.TestCase):
    """Test result formatting for context injection."""

    def test_empty_results(self):
        self.assertEqual(_format_results([]), "")

    def test_single_result(self):
        results = [{"content": "Billing uses Stripe API", "fitness_score": 0.85, "dimension": 3}]
        formatted = _format_results(results)
        self.assertIn("KONO Memory", formatted)
        self.assertIn("Billing uses Stripe API", formatted)
        self.assertIn("fitness=0.85", formatted)

    def test_permanent_memory_flagged(self):
        results = [{"content": "Safety gate required", "fitness_score": 0.95, "is_permanent": True}]
        formatted = _format_results(results)
        self.assertIn("permanent", formatted)

    def test_multiple_results(self):
        results = [
            {"content": "First memory", "fitness_score": 0.9},
            {"content": "Second memory", "fitness_score": 0.7},
            {"content": "Third memory", "fitness_score": 0.5},
        ]
        formatted = _format_results(results)
        self.assertIn("1.", formatted)
        self.assertIn("2.", formatted)
        self.assertIn("3.", formatted)

    def test_truncation_under_budget(self):
        results = [{"content": "x" * 5000, "fitness_score": 0.5}]
        formatted = _format_results(results)
        self.assertLessEqual(len(formatted), _MAX_CONTEXT_CHARS)

    def test_empty_content_skipped(self):
        results = [
            {"content": "", "fitness_score": 0.9},
            {"content": "Real content", "fitness_score": 0.8},
        ]
        formatted = _format_results(results)
        self.assertIn("Real content", formatted)

    def test_text_field_fallback(self):
        """KONO results may use 'text' instead of 'content'."""
        results = [{"text": "Fallback text field", "fitness_score": 0.6}]
        formatted = _format_results(results)
        self.assertIn("Fallback text field", formatted)


class TestTimeoutProtection(unittest.TestCase):
    """Test that KONO recall respects timeout."""

    @patch("lib.hal.kono_bridge._kono_available", return_value=True)
    @patch("lib.hal.kono_bridge._execute_recall")
    def test_timeout_returns_empty(self, mock_execute, _mock_avail):
        """Simulate a slow KONO query that exceeds timeout."""
        import time

        def slow_recall(*args, **kwargs):
            time.sleep(5)
            return "too late"

        mock_execute.side_effect = slow_recall
        clear_cache()
        result = recall_for_query("slow query", timeout_s=0.1)
        self.assertEqual(result, "")

    @patch("lib.hal.kono_bridge._kono_available", return_value=True)
    @patch("lib.hal.kono_bridge._execute_recall", side_effect=ImportError("no apsw"))
    def test_import_error_returns_empty(self, _mock_exec, _mock_avail):
        clear_cache()
        result = recall_for_query("query that fails to import")
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
