"""Deterministic tool-output pruning for the verbose `lookup_order` response.

The "Tool Context Pruning" pattern: application-side filtering
of a verbose tool result so only the fields needed for the immediate decision survive
into context. For return/refund reasoning, exactly five fields matter — order identity,
when it was placed, what it cost, whether it shipped, and the return-window deadline.

# TODO (Exercise 1): Replace the placeholder docstring section below with a
# justification of each of the 5 kept fields. The justifications are reviewed:
# a reviewer reads them and must agree they map to the return/refund
# decision, not to "looks important" or "might be useful later". One short
# sentence per field is enough.
#
# Why each kept field is the only one that matters for return/refund reasoning:
#   - order_id: identifies the order so the decision can be tied back to the
#     specific record and communicated to the customer.
#   - order_date: anchors the return-window calculation, since eligibility is
#     computed relative to when the order was placed.
#   - order_total_usd: caps how much can be refunded; the agent cannot approve
#     a refund larger than what was paid.
#   - fulfillment_status: determines whether a cancel (not yet shipped) or a
#     return (already delivered) is the applicable path.
#   - return_eligible_until: the deadline the agent compares against today's
#     date to decide whether the return window is still open.

Implementation: deterministic field selection (no LLM call). The pruner has no
`anthropic` import — enforced by an AST audit.
"""
from __future__ import annotations

# TODO (Exercise 1): Replace with the exact 5-field tuple, in OUTPUT ORDER.
# These are the only fields the pruner returns; everything else in the raw
# response is dropped. The output dict preserves this declaration order.
#
# The 5 fields: order_id, order_date, order_total_usd, fulfillment_status,
# return_eligible_until — chosen because they are the *only* fields needed for
# the agent's return/refund decision.
KEPT_FIELDS: tuple[str, ...] = (
    "order_id",
    "order_date",
    "order_total_usd",
    "fulfillment_status",
    "return_eligible_until",
)


class PrunerMissingFieldError(KeyError):
    """Raised when the raw tool response is missing one of the required kept fields."""


def prune_lookup_order(raw: dict) -> dict:
    # TODO (Exercise 1): Implement deterministic field selection.
    #
    # 1. Check that every name in KEPT_FIELDS is present as a key in `raw`.
    #    If any are missing, raise PrunerMissingFieldError with a message
    #    that lists the missing field names — the agent needs to *notice*
    #    the upstream tool returned an incomplete record, not silently
    #    propagate it.
    #
    # 2. Return a new dict containing exactly the KEPT_FIELDS, in their
    #    declaration order. (Preserving the order is part of the contract;
    #    tests/test_pruner.py asserts on it.)
    #
    # Do NOT add an `anthropic` import here — the pruner is deterministic by
    # design. The AST audit will flag any LLM-driven implementation.
    missing = [field for field in KEPT_FIELDS if field not in raw]
    if missing:
        raise PrunerMissingFieldError(
            f"lookup_order response is missing required kept fields: {missing}"
        )
    return {field: raw[field] for field in KEPT_FIELDS}
