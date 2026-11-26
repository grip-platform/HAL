"""Signed decision gate shared by the Claude wire and OpenAI rails."""


def sign_failover_decision(provider: str, model: str, billing: str,
                           session_id: str | None) -> bool:
    """A target may receive a request only after its decision row is written.

    True only when the row is written. A signer or store that raises has not
    signed, so that is False as well: both rails then record ``unsigned``
    and send nothing, whatever their own error handling would have done.
    """
    from lib.hal.gateway_witness import append_thread_row

    try:
        result = append_thread_row({'provider': provider, 'model': model, 'status': None},
                                   row_class='failover', route='inference',
                                   session_id=session_id, billing=billing)
    except Exception:  # noqa: BLE001 - no signed row means no move, for any cause
        return False
    return bool(getattr(result, 'written', False))
