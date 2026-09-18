"""Case-facts extraction into a persistent block at the top of context.

Extraction is LLM-driven: one Claude call against the full transcript that returns
strict JSON for the 12 required fields. This is commonly called a *scratchpad* — same
concept, different word: a dense structured block that survives compression and is
placed at the top boundary of context so the model can recover transactional facts
without scanning thousands of tokens of narrative.

Missing-field behavior raises `CaseFactExtractionError` listing the gaps — silent
null-fill is forbidden.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from retail_context.client import complete_with_system, get_model
from retail_context.transcript import Transcript

# TODO (Exercise 2): Replace with the 12-field contract for the case-facts block.
# Order matters — the to_markdown() rendering uses this order so the block is
# reviewer-readable and the model can re-find a value by name. Each name appears
# verbatim in the extraction system prompt as the JSON key the LLM must return.
#
# Required fields (12 total): customer_id; refund_order_id, refund_amount_usd,
# refund_status; subscription_id, subscription_plan, subscription_cancel_reason,
# subscription_status; active_payment_method_last4, new_payment_method_last4,
# payment_update_failure_code, payment_update_status.
REQUIRED_FIELDS: tuple[str, ...] = (
    "customer_id",
    "refund_order_id",
    "refund_amount_usd",
    "refund_status",
    "subscription_id",
    "subscription_plan",
    "subscription_cancel_reason",
    "subscription_status",
    "active_payment_method_last4",
    "new_payment_method_last4",
    "payment_update_failure_code",
    "payment_update_status",
)


# TODO (Exercise 2): Replace with a dataclass that carries the 12 fields above
# with explicit Python types (str for IDs / status tokens / last4 strings,
# float for refund_amount_usd). Then implement to_markdown(self) -> str so the
# block renders with a level-1 `# Case Facts` header and the three subgroups
# (Customer / Refund (resolved) / Subscription (resolved) / Payment update
# (active)) as bold inline labels. The rendering is part of the contract:
# fixed key order, Markdown headers, reviewer-readable.
@dataclass
class CaseFacts:
    customer_id: str
    refund_order_id: str
    refund_amount_usd: float
    refund_status: str
    subscription_id: str
    subscription_plan: str
    subscription_cancel_reason: str
    subscription_status: str
    active_payment_method_last4: str
    new_payment_method_last4: str
    payment_update_failure_code: str
    payment_update_status: str

    def to_markdown(self) -> str:
        lines = [
            "# Case Facts",
            "",
            f"**Customer**: {self.customer_id}",
            "",
            "**Refund (resolved)**",
            f"- order_id: {self.refund_order_id}",
            f"- amount_usd: {self.refund_amount_usd}",
            f"- status: {self.refund_status}",
            "",
            "**Subscription (resolved)**",
            f"- subscription_id: {self.subscription_id}",
            f"- plan: {self.subscription_plan}",
            f"- cancel_reason: {self.subscription_cancel_reason}",
            f"- status: {self.subscription_status}",
            "",
            "**Payment update (active)**",
            f"- active_payment_method_last4: {self.active_payment_method_last4}",
            f"- new_payment_method_last4: {self.new_payment_method_last4}",
            f"- failure_code: {self.payment_update_failure_code}",
            f"- status: {self.payment_update_status}",
        ]
        return "\n".join(lines)


class CaseFactExtractionError(ValueError):
    def __init__(self, missing: list[str], raw: dict[str, Any]):
        super().__init__(f"case-facts extraction missing required fields: {missing}")
        self.missing = missing
        self.raw = raw


# TODO (Exercise 2): Replace with the system prompt that drives the extraction.
# The prompt must require Claude to return EXACTLY one JSON object with the
# 12 REQUIRED_FIELDS as keys, with the right types (refund_amount_usd is a
# number, last4 are zero-padded strings, status tokens preserved verbatim).
# Missing fields are null — DO NOT invent. Output is JSON only — no prose,
# no markdown, no code fences. The prompt is reviewed for its strict-schema
# intent; the reviewer reads it to decide whether following it would reliably
# produce a parseable JSON with every required field.
_SYSTEM_PROMPT = """You extract structured case facts from a customer support transcript.

Return EXACTLY one JSON object with these keys and nothing else — no prose, no
markdown, no code fences:

  customer_id, refund_order_id, refund_amount_usd, refund_status,
  subscription_id, subscription_plan, subscription_cancel_reason,
  subscription_status, active_payment_method_last4, new_payment_method_last4,
  payment_update_failure_code, payment_update_status

Rules:
- refund_amount_usd must be a JSON number (not a string).
- last4 fields are zero-padded 4-digit strings (e.g. "0042").
- Status/reason/plan fields are the exact tokens used in the transcript
  (e.g. "cancelled_with_prorated_refund"), not paraphrased natural language.
- If a field's value cannot be found in the transcript, its value is JSON
  null. Never invent or guess a value.
- Output JSON only. No explanation before or after it."""


def _parse_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.rstrip().endswith("```"):
            raw = raw.rsplit("```", 1)[0]
    return json.loads(raw)


def extract(
    transcript: Transcript,
    *,
    model: str | None = None,
    log_path: Path | None = None,
) -> CaseFacts:
    # TODO (Exercise 2): Run the extraction call and validate the result.
    #
    # 1. Build the user message: f"Transcript:\n\n{transcript.full_text}".
    #
    # 2. Call complete_with_system(_SYSTEM_PROMPT, user, model=model,
    #    max_tokens=2048). It returns (text, input_tokens, output_tokens).
    #
    # 3. Parse the response text with _parse_json (handles a stray ``` fence).
    #
    # 4. If a `log_path` was given, write a JSON log containing the model
    #    name, the input/output token counts, and the raw parsed dict. This
    #    is the `case_facts_call.json` artifact the reviewer audits.
    #
    # 5. Validate that every name in REQUIRED_FIELDS is present in the parsed
    #    dict and is not None and not the empty string. If anything is
    #    missing, raise CaseFactExtractionError(missing=..., raw=...) — DO
    #    NOT silently fill with null.
    #
    # 6. Construct and return a CaseFacts. Cast types explicitly:
    #    str(...) for ID/status fields, float(...) for refund_amount_usd.
    user = f"Transcript:\n\n{transcript.full_text}"
    target_model = model or get_model()
    text, input_tokens, output_tokens = complete_with_system(
        _SYSTEM_PROMPT, user, model=model, max_tokens=2048
    )
    parsed = _parse_json(text)

    if log_path is not None:
        log_path.write_text(
            json.dumps(
                {
                    "model": target_model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "raw": parsed,
                },
                indent=2,
            )
        )

    missing = [
        name
        for name in REQUIRED_FIELDS
        if parsed.get(name) is None or parsed.get(name) == ""
    ]
    if missing:
        raise CaseFactExtractionError(missing=missing, raw=parsed)

    return CaseFacts(
        customer_id=str(parsed["customer_id"]),
        refund_order_id=str(parsed["refund_order_id"]),
        refund_amount_usd=float(parsed["refund_amount_usd"]),
        refund_status=str(parsed["refund_status"]),
        subscription_id=str(parsed["subscription_id"]),
        subscription_plan=str(parsed["subscription_plan"]),
        subscription_cancel_reason=str(parsed["subscription_cancel_reason"]),
        subscription_status=str(parsed["subscription_status"]),
        active_payment_method_last4=str(parsed["active_payment_method_last4"]),
        new_payment_method_last4=str(parsed["new_payment_method_last4"]),
        payment_update_failure_code=str(parsed["payment_update_failure_code"]),
        payment_update_status=str(parsed["payment_update_status"]),
    )


def to_dict(facts: CaseFacts) -> dict[str, Any]:
    return asdict(facts)
