"""The `tool_call` event table row must agree with the tool loop beneath it (HAL#557).

happi.md's event table is the contract (axiom 4). Its `tool_call` row said
"Invoking a tool" while the Tool-loop paragraph three lines below said the
opposite -- "no runtime executes it -- the caller runs the tool". A consumer
reading the table alone could not tell whether the event announced a request
or an invocation that had already happened. The adapter settles it: in
`lib/hal/happi/providers/openai_compat.py::_terminate`, `tool_call` is emitted
only when `finish_reason == "tool_calls"` -- the model has finished ASKING --
and nothing in HAL runs the tool.

The round-trip gate (`build-happi.py --check`) structurally cannot catch this.
It proves happi.md was assembled from happi-sources/, and in this incident both
sides carried the same wrong sentence. These checks assert the property
round-trip cannot: that two places describing one event describe it the SAME way.

`test_tool_result_row_is_readable` is the Goodhart control -- it fails if the
table extractor stops reading the real table, so the other checks cannot pass
vacuously.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HAPPI_MD = REPO / "happi.md"
EVENTS_PY = REPO / "lib" / "hal" / "happi" / "events.py"

# Words that assert the invocation already happened, with no actor named.
# The drift class HAL#557 closed: a Meaning cell that reads as HAL doing it.
_INVOCATION_WORDS = ("invoking", "invokes", "invoked")


def _events_table(text: str) -> str:
    """The event table section only -- other tables in the spec are not it."""
    start = text.index("## Events (stdout)")
    return text[start : text.index("\n---", start)]


def _row_meaning(text: str, event_type: str) -> str:
    """The Meaning cell of the 3-column table row for `event_type`.

    Cells are stripped, so an indented docstring quote and a
    column-padded spec row compare equal on content."""
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 3 and cells[0].startswith(f"`{event_type}`"):
            return cells[2]
    raise AssertionError(f"no event-table row for {event_type!r}")


def _spec_meaning(event_type: str) -> str:
    """The contract's own wording, read from the assembled happi.md."""
    return _row_meaning(_events_table(HAPPI_MD.read_text()), event_type)


def test_tool_result_row_is_readable():
    """Goodhart control: the extractor reads the real table, not nothing."""
    assert _spec_meaning("tool_result") == "Tool returned"


def test_tool_call_row_names_who_asks_and_who_runs():
    """The row must resolve request-vs-invocation on its own, per HAL#557."""
    meaning = _spec_meaning("tool_call")
    lowered = meaning.lower()
    bare = [w for w in _INVOCATION_WORDS if w in lowered]
    assert not bare, (
        f"tool_call row reads as an invocation ({bare}): {meaning!r}. "
        "No HAPPI runtime executes a tool; say who asked and who runs it."
    )
    assert "ask" in lowered, f"tool_call row must say the model ASKED: {meaning!r}"
    assert "caller" in lowered, f"tool_call row must name the CALLER: {meaning!r}"


def test_tool_loop_paragraph_still_places_execution_with_the_caller():
    """The other half of the agreement -- fixing the row by deleting this fires."""
    body = _events_table(HAPPI_MD.read_text())
    assert "no runtime executes it" in body
    assert "the caller runs the tool" in body


def test_events_py_quotes_the_contract_row_verbatim():
    """`emit_tool_call`'s docstring quotes the table row and says the table is
    the contract. A quote that silently goes stale is how HAL#557 reached two
    surfaces; this pins the quote to what happi.md actually says."""
    quoted = _row_meaning(EVENTS_PY.read_text(), "tool_call")
    assert quoted == _spec_meaning("tool_call"), (
        "lib/hal/happi/events.py quotes a tool_call row that happi.md no "
        "longer carries -- the docstring drifted from the contract it cites."
    )
