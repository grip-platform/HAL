"""Tests for HAL response streamer."""
import io
import os
import unittest
from unittest import mock

from lib.hal.response_streamer import ResponseStreamer


class _FakeTTY(io.StringIO):
    """StringIO that reports itself as a TTY so render_to_stdout takes the streaming path."""

    def isatty(self) -> bool:
        return True


class TestResponseStreamer(unittest.TestCase):
    def test_stream_words(self):
        chunks = ResponseStreamer().stream("Hello world foo bar", chunk_size=2)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(chunks[-1].is_final)

    def test_empty(self):
        chunks = ResponseStreamer().stream("")
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].is_final)

    def test_stream_chars(self):
        chunks = ResponseStreamer().stream_chars("Hello world", chars_per_chunk=5)
        self.assertGreater(len(chunks), 1)

    def test_indices(self):
        chunks = ResponseStreamer().stream("a b c d e f", chunk_size=2)
        for i, c in enumerate(chunks):
            self.assertEqual(c.index, i)


class TestRenderToStdout(unittest.TestCase):
    """render_to_stdout — the W1 cosmetic streaming path."""

    def test_non_tty_falls_back_to_single_write(self):
        sink = io.StringIO()  # no isatty() override -> False
        mock_sink = mock.Mock(wraps=sink)
        mock_sink.isatty.return_value = False
        text = "The quick brown fox jumps over the lazy dog."
        writes = ResponseStreamer().render_to_stdout(text, stream=mock_sink, delay_ms=0)
        self.assertEqual(writes, 1)
        self.assertEqual(mock_sink.write.call_count, 1)
        self.assertEqual(mock_sink.write.call_args_list[0].args[0], text)

    def test_tty_emits_multiple_writes_with_flush(self):
        sink = _FakeTTY()
        mock_sink = mock.Mock(wraps=sink)
        mock_sink.isatty.return_value = True
        text = "A" * 100  # 5 chunks at chars_per_chunk=20
        writes = ResponseStreamer().render_to_stdout(
            text, stream=mock_sink, chars_per_chunk=20, delay_ms=0, caret=""
        )
        self.assertGreater(writes, 1)
        self.assertGreater(mock_sink.write.call_count, 1)
        self.assertGreater(mock_sink.flush.call_count, 1)
        # The reassembled text contains the full input (caret is empty).
        self.assertIn("A" * 100, sink.getvalue())

    def test_tty_paints_and_erases_caret(self):
        sink = _FakeTTY()
        text = "hello world"  # short, will hit caret path between chunks
        ResponseStreamer().render_to_stdout(
            text, stream=sink, chars_per_chunk=3, delay_ms=0,
            caret="|",  # single-char caret so \b\s\b aligns
        )
        output = sink.getvalue()
        # Caret paint appears at least once.
        self.assertIn("|", output)
        # Erase sequence (backspace + space + backspace) appears at least once.
        self.assertIn("\b \b", output)
        # Final user-visible text is still present after cursor tricks.
        # Strip cursor control characters to recover the content.
        stripped = output.replace("|", "").replace("\b \b", "")
        self.assertEqual(stripped, text)

    def test_caret_suppressed_when_chunk_ends_with_newline(self):
        sink = _FakeTTY()
        # chars_per_chunk=3 splits "abc\ndefghi" into ["abc", "\nde", "fgh", "i"].
        # Chunk 0 ("abc") is non-final and does NOT end in \n -> caret painted.
        # Chunk 1 ("\nde") is non-final and does NOT end in \n -> caret painted.
        # A chunk that DOES end in \n must not paint caret, because the cursor
        # is then at col 0 where \b is undefined and would leave visible garbage.
        ResponseStreamer().render_to_stdout(
            "ab\ncd", stream=sink, chars_per_chunk=3, delay_ms=0, caret="|"
        )
        # Chunks at chars_per_chunk=3: ["ab\n", "cd"]. Chunk 0 ends with "\n"
        # (non-final) -> caret suppressed. Chunk 1 is final -> no caret.
        self.assertNotIn("|", sink.getvalue())
        self.assertEqual(sink.getvalue(), "ab\ncd")

    def test_hal_no_stream_env_forces_atomic_write(self):
        sink = _FakeTTY()
        mock_sink = mock.Mock(wraps=sink)
        mock_sink.isatty.return_value = True
        with mock.patch.dict(os.environ, {"HAL_NO_STREAM": "1"}):
            writes = ResponseStreamer().render_to_stdout("abcdefg" * 20, stream=mock_sink, delay_ms=0)
        self.assertEqual(writes, 1)
        self.assertEqual(mock_sink.write.call_count, 1)

    def test_empty_text_writes_nothing(self):
        sink = _FakeTTY()
        writes = ResponseStreamer().render_to_stdout("", stream=sink, delay_ms=0)
        self.assertEqual(writes, 0)
        self.assertEqual(sink.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
