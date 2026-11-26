# HAPPI queue demo

Small polyglot document used by `test_happi_runner.py` to exercise the
γ-2 queue driver end-to-end against a realistic polyglot structure
(prose + fenced block).

## Run queue

```ndjson
{"v":"happi/1.0","id":"demo-1","cmd":"echo","args":["hello","queue"]}
{"v":"happi/1.0","id":"demo-2","cmd":"noop"}
{"v":"happi/1.0","id":"demo-3","cmd":"version"}
```

## Also some prose

Prose after the queue block must not be consumed by the parser.
