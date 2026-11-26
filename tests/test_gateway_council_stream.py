"""The council relay turns the VPS's single JSON reply into a stream a client can parse.

A streaming client must always get a finish_reason and [DONE] on a good reply, and an
error it can read on a bad one, never an empty answer that looks finished.
"""
import io
import json
import unittest
from unittest import mock

from lib.hal import gateway_openai as gw


class FakeHandler:
    def __init__(self):
        self.wfile = io.BytesIO()
        self.status = None
        self.headers = {}

    def send_response(self, status):
        self.status = status

    def send_header(self, name, value):
        self.headers[name] = value

    def end_headers(self):
        pass


def _events(handler):
    frames = [f for f in handler.wfile.getvalue().decode().split("\n\n") if f]
    assert frames[-1] == "data: [DONE]"
    return [json.loads(line[len("data: "):])
            for frame in frames[:-1] for line in frame.splitlines()
            if line.startswith("data: ")]


def _stream(body):
    handler = FakeHandler()
    gw._relay_council_stream(handler, body, "council", "abc", "leaf")
    return handler


class CouncilStreamTest(unittest.TestCase):
    def test_a_good_reply_carries_the_text_and_a_finish_reason(self):
        body = json.dumps({"choices": [{"message": {"content": "verdict"}}]}).encode()
        handler = _stream(body)
        events = _events(handler)
        self.assertEqual(handler.headers["Content-Type"], "text/event-stream")
        self.assertEqual(events[0]["choices"][0]["delta"], {"role": "assistant"})
        self.assertEqual(events[1]["choices"][0]["delta"], {"content": "verdict"})
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "stop")

    def test_a_truncated_reply_keeps_its_finish_reason(self):
        body = json.dumps({"choices": [{"message": {"content": "part"}, "finish_reason": "length"}]}).encode()
        self.assertEqual(_events(_stream(body))[-1]["choices"][0]["finish_reason"], "length")

    def test_the_seal_headers_pass_through(self):
        handler = _stream(json.dumps({"choices": [{"message": {"content": "v"}}]}).encode())
        self.assertEqual(handler.headers["X-Grip-Council-Sha"], "abc")
        self.assertEqual(handler.headers["X-Grip-Idr-Leaf"], "leaf")

    def _assert_reports_error(self, body, wanted="unreadable"):
        text = _stream(body).wfile.getvalue().decode()
        self.assertIn(wanted, text)
        self.assertNotIn('"finish_reason": "stop"', text)
        self.assertTrue(text.endswith("data: [DONE]\n\n"))

    def test_invalid_json_is_reported(self):
        self._assert_reports_error(b"<html>bad gateway</html>")

    def test_a_first_choice_that_is_not_an_object_is_reported(self):
        self._assert_reports_error(json.dumps({"choices": ["oops"]}).encode())

    def test_a_reply_with_no_text_is_reported(self):
        for content in (None, "", 7):
            self._assert_reports_error(json.dumps({"choices": [{"message": {"content": content}}]}).encode())

    def test_an_empty_choices_list_is_reported(self):
        self._assert_reports_error(json.dumps({"choices": []}).encode())

    def test_the_vps_error_message_reaches_the_client(self):
        self._assert_reports_error(json.dumps({"error": {"message": "no quorum"}}).encode(), "no quorum")

    def test_a_non_string_error_message_is_reported(self):
        for message in (7, ["busy"], {"reason": "busy"}):
            with self.subTest(message=message):
                handler = _stream(json.dumps({"error": {"message": message}}).encode())
                events = _events(handler)
                self.assertEqual(events[0]["error"]["message"], str(message))
                self.assertEqual(len(events), 1)


class CouncilJsonTest(unittest.TestCase):
    def test_a_vps_error_keeps_its_status_and_body(self):
        handler = FakeHandler()
        gw._relay_council_json(handler, b'{"error":"bad"}', 502, "council", "", "")
        self.assertEqual(handler.status, 502)
        self.assertEqual(handler.wfile.getvalue(), b'{"error":"bad"}')
        self.assertNotIn("X-Grip-Council-Sha", handler.headers)


class CouncilRouteTest(unittest.TestCase):
    """Which branch a request takes, with the VPS call faked."""

    def _route(self, payload, *, reply=None, error=None):
        class Reply:
            status = 200
            headers = {"X-Grip-Council-Sha": "s", "X-Grip-Idr-Leaf": "l"}

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return reply

        handler = FakeHandler()
        target = "urllib.request.urlopen"
        effect = {"side_effect": error} if error else {"return_value": Reply()}
        with mock.patch(target, **effect) as call, \
                mock.patch.object(gw, "_witness_chat"), \
                mock.patch.object(gw, "_relay_council_failopen") as failopen:
            gw._relay_council(handler, payload, "council", 0.0)
        return handler, call, failopen

    def test_the_vps_is_always_asked_for_one_non_streaming_reply(self):
        reply = json.dumps({"choices": [{"message": {"content": "v"}}]}).encode()
        _, call, _ = self._route({"stream": True, "messages": []}, reply=reply)
        sent = json.loads(call.call_args[0][0].data)
        self.assertIs(sent["stream"], False)

    def test_stream_only_options_never_reach_the_batch_request(self):
        reply = json.dumps({"choices": [{"message": {"content": "v"}}]}).encode()
        _, call, _ = self._route({"stream": True, "stream_options": {"include_usage": True}, "messages": []}, reply=reply)
        self.assertNotIn("stream_options", json.loads(call.call_args[0][0].data))

    def test_usage_is_emitted_only_when_requested_and_available(self):
        usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        for options in (None, {"include_usage": False}, {"include_usage": True}):
            for available in (False, True):
                with self.subTest(options=options, available=available):
                    reply = {"choices": [{"message": {"content": "v"}}]}
                    if available:
                        reply["usage"] = usage
                    payload = {"stream": True, "messages": []}
                    if options is not None:
                        payload["stream_options"] = options
                    handler, _, _ = self._route(payload, reply=json.dumps(reply).encode())
                    events = _events(handler)
                    if available and options and options.get("include_usage"):
                        self.assertEqual(events[-1]["choices"], [])
                        self.assertEqual(events[-1]["usage"], usage)
                        self.assertEqual(events[-2]["choices"][0]["finish_reason"], "stop")
                        for key in ("id", "object", "created", "model"):
                            self.assertEqual(events[-1][key], events[0][key])
                    else:
                        self.assertEqual(len(events), 3)
                        self.assertFalse(any("usage" in event for event in events))

    def test_non_object_stream_options_do_not_break_the_stream(self):
        reply = json.dumps({
            "choices": [{"message": {"content": "v"}}],
            "usage": {"total_tokens": 5},
        }).encode()
        for options in (True, "include_usage", ["include_usage"], 1, False, None, [], "", 0):
            with self.subTest(options=options):
                handler, call, failopen = self._route(
                    {"stream": True, "stream_options": options}, reply=reply)
                events = _events(handler)
                self.assertEqual(handler.status, 200)
                self.assertEqual(events[-1]["choices"][0]["finish_reason"], "stop")
                self.assertFalse(any("usage" in event for event in events))
                self.assertNotIn("stream_options", json.loads(call.call_args[0][0].data))
                failopen.assert_not_called()

    def test_http_errors_keep_retry_headers_and_use_openai_status(self):
        import email.message
        import urllib.error

        for status in (400, 429, 503):
            for stream in (False, True):
                with self.subTest(status=status, stream=stream):
                    headers = email.message.Message()
                    headers["Retry-After"] = "12"
                    headers["retry-after-ms"] = "12000"
                    headers["Content-Length"] = "999"
                    headers["Connection"] = "keep-alive"
                    body = b'{"error":"busy"}'
                    error = urllib.error.HTTPError(
                        "http://vps", status, "busy", headers, io.BytesIO(body))
                    handler, _, failopen = self._route({"stream": stream}, error=error)
                    self.assertEqual(handler.status, 503 if status == 429 else status)
                    self.assertEqual(handler.headers["Retry-After"], "12")
                    self.assertEqual(handler.headers["retry-after-ms"], "12000")
                    self.assertEqual(handler.headers["Content-Length"], str(len(body)))
                    self.assertEqual(handler.headers["Connection"], "close")
                    self.assertEqual(handler.wfile.getvalue(), body)
                    failopen.assert_not_called()

    def test_a_malformed_vps_url_falls_back(self):
        failopen = mock.Mock()
        with mock.patch.dict("os.environ", {"GRIP_COUNCIL_VPS_URL": "not a url"}), \
                mock.patch.object(gw, "_witness_chat"), \
                mock.patch.object(gw, "_relay_council_failopen", failopen):
            gw._relay_council(FakeHandler(), {"messages": []}, "council", 0.0)
        failopen.assert_called_once()

    def test_a_vps_error_whose_body_cannot_be_read_keeps_its_status(self):
        import email.message
        import urllib.error

        class Broken(io.BytesIO):
            def read(self, *args):
                raise TimeoutError("reset")

        error = urllib.error.HTTPError("http://vps", 503, "busy", email.message.Message(), Broken())
        handler, _, failopen = self._route({"messages": []}, error=error)
        self.assertEqual(handler.status, 503)
        self.assertIn(b"could not be read", handler.wfile.getvalue())
        failopen.assert_not_called()

    def test_a_streaming_request_gets_a_stream_and_a_plain_one_gets_json(self):
        reply = json.dumps({"choices": [{"message": {"content": "v"}}]}).encode()
        streamed, _, _ = self._route({"stream": True, "messages": []}, reply=reply)
        plain, _, _ = self._route({"messages": []}, reply=reply)
        self.assertEqual(streamed.headers["Content-Type"], "text/event-stream")
        self.assertEqual(plain.headers["Content-Type"], "application/json")
        self.assertEqual(plain.wfile.getvalue(), reply)
        for handler in (streamed, plain):
            self.assertEqual(handler.status, 200)
            self.assertEqual(handler.headers["X-Grip-Council-Sha"], "s")
            self.assertEqual(handler.headers["X-Grip-Idr-Leaf"], "l")
            self.assertNotIn("Retry-After", handler.headers)

    def test_a_vps_http_error_is_relayed_and_never_falls_back(self):
        import email.message
        import urllib.error
        error = urllib.error.HTTPError(
            "http://vps", 503, "busy", email.message.Message(), io.BytesIO(b'{"error":"busy"}'))
        handler, _, failopen = self._route({"stream": True, "messages": []}, error=error)
        self.assertEqual(handler.status, 503)
        self.assertEqual(handler.wfile.getvalue(), b'{"error":"busy"}')
        failopen.assert_not_called()

    def test_an_unreachable_vps_falls_back(self):
        _, _, failopen = self._route({"messages": []}, error=OSError("no route"))
        failopen.assert_called_once()


if __name__ == "__main__":
    unittest.main()
