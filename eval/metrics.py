"""Correctness metrics for invoice extraction.

Pure functions over (prediction, ground_truth, ocr_text). No I/O — the eval
runner (eval/runner.py) handles loading + writing results.

The six metrics from the spec:
1. Schema validity rate           — does the model's raw JSON output parse + validate
2. Per-field exact-match accuracy — with field-specific normalization
3. Numerical tolerance match      — total_amount within 1% counts as match
4. Date normalization match       — different formats parsed to the same date count
5. Record-level accuracy          — every scored field correct for that record
6. Hallucination rate             — non-null predictions whose value isn't in OCR text

Sparse-GT contract: expected_json contains only the fields we actually have
ground truth for (SROIE has no invoice_number / currency / line_items). The
record's accuracy is judged on those keys only. Predictions for fields outside
GT are scored only by the hallucination metric (must be grounded in OCR text).

Caveat on grounding: the hallucination check uses substring matching after
normalization. For numerical fields, small values can match by coincidence
(e.g. predicted total=99.0 is "grounded" by a vendor name like "99 SPEED MART").
This metric is therefore an upper bound on faithfulness — a model can be more
hallucinated than it appears, but not less. Acceptable for Part A; can be
tightened later with context-aware matching (number near "TOTAL" keyword, etc).
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any

from dateutil import parser as date_parser
from pydantic import ValidationError

from service.schema import Invoice

SCALAR_FIELDS = ("vendor_name", "invoice_number", "invoice_date", "total_amount", "currency")

# ---------- normalization ----------

# Trailing parenthetical containing at least one digit, e.g. " (519537-X)" or "(123)".
# Used to strip company registration codes that GPT-4o-mini sometimes appends
# to vendor_name even though SROIE's ground truth doesn't include them.
_TRAILING_REG_CODE_RE = re.compile(r"\s*\([^)]*\d[^)]*\)\s*$")
_WS_RE = re.compile(r"\s+")


def _normalize_str(s: str) -> str:
    """Casefold + strip + collapse internal whitespace."""
    return _WS_RE.sub(" ", s.strip().casefold())


def _normalize_vendor(s: str) -> str:
    """Vendor name normalization — strip trailing reg codes, then standard string norm."""
    s = _TRAILING_REG_CODE_RE.sub("", s)
    return _normalize_str(s)


# ---------- per-field comparators ----------

def compare_vendor_name(pred: Any, expected: Any) -> bool:
    if pred is None and expected is None:
        return True
    if pred is None or expected is None:
        return False
    return _normalize_vendor(str(pred)) == _normalize_vendor(str(expected))


def compare_invoice_number(pred: Any, expected: Any) -> bool:
    if pred is None and expected is None:
        return True
    if pred is None or expected is None:
        return False
    # IDs may have stylistic spacing differences — collapse all whitespace.
    return _normalize_str(str(pred)).replace(" ", "") == _normalize_str(str(expected)).replace(" ", "")


def compare_invoice_date(pred: Any, expected: Any) -> bool:
    """Both parsed with dateutil → compare on the date object. Handles ISO drift,
    trailing time components, alternate format strings."""
    if pred is None and expected is None:
        return True
    if pred is None or expected is None:
        return False
    try:
        pd = date_parser.parse(str(pred), dayfirst=True, fuzzy=True).date()
        ed = date_parser.parse(str(expected), dayfirst=True, fuzzy=True).date()
    except (ValueError, TypeError, OverflowError):
        return False
    return pd == ed


def compare_total_amount(pred: Any, expected: Any, *, tol: float = 0.01) -> bool:
    """Match if |pred - expected| / |expected| <= tol. Default 1%.

    Relative-to-expected (conventional) rather than relative-to-max — keeps the
    interpretation of "within 1% of the true total" intuitive.
    """
    if pred is None and expected is None:
        return True
    if pred is None or expected is None:
        return False
    try:
        p = float(pred)
        e = float(expected)
    except (TypeError, ValueError):
        return False
    if e == 0:
        return p == 0
    return abs(p - e) / abs(e) <= tol


def compare_currency(pred: Any, expected: Any) -> bool:
    if pred is None and expected is None:
        return True
    if pred is None or expected is None:
        return False
    return str(pred).strip().upper() == str(expected).strip().upper()


COMPARATORS = {
    "vendor_name": compare_vendor_name,
    "invoice_number": compare_invoice_number,
    "invoice_date": compare_invoice_date,
    "total_amount": compare_total_amount,
    "currency": compare_currency,
}


def field_match(field_name: str, pred: Any, expected: Any) -> bool:
    """Dispatch by name. Unknown field names → False (defensive)."""
    cmp = COMPARATORS.get(field_name)
    if cmp is None:
        return False
    return cmp(pred, expected)


# ---------- schema validity ----------

def is_schema_valid(raw: str | dict[str, Any] | None) -> bool:
    """True iff `raw` (string or dict) parses as JSON and validates as Invoice."""
    if raw is None:
        return False
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return False
    else:
        data = raw
    try:
        Invoice.model_validate(data)
    except ValidationError:
        return False
    return True


# ---------- hallucination grounding ----------

# ISO 4217 codes mapped to symbols / synonyms commonly seen on receipts.
# Used to decide whether a predicted currency is "supported" by the OCR text
# (predicted MYR is grounded if "RM" or "MYR" appears anywhere in the text).
CURRENCY_TOKENS: dict[str, list[str]] = {
    "USD": ["USD", "US$", "$"],
    "MYR": ["MYR", "RM"],
    "EUR": ["EUR", "€"],
    "GBP": ["GBP", "£"],
    "SGD": ["SGD", "S$"],
    "JPY": ["JPY", "¥"],
    "CNY": ["CNY", "RMB", "¥"],
    "INR": ["INR", "₹", "RS", "RS."],
    "AUD": ["AUD", "A$"],
    "CAD": ["CAD", "C$"],
}


def _has_substring_normalized(text: str, needle: str) -> bool:
    return _normalize_str(needle) in _normalize_str(text)


def is_grounded(field_name: str, value: Any, ocr_text: str) -> bool:
    """Is `value` (predicted for `field_name`) findable in `ocr_text`?

    Null values are vacuously grounded (the model honestly said "I don't know").
    Non-null values are grounded iff some normalized form of the value appears
    in the OCR text — exact rules per field.
    """
    if value is None:
        return True
    if field_name == "vendor_name":
        v = str(value)
        # Try the verbatim string AND the reg-code-stripped form.
        return _has_substring_normalized(ocr_text, v) or _has_substring_normalized(
            ocr_text, _normalize_vendor(v)
        )
    if field_name == "invoice_number":
        # IDs often have stylistic spacing — strip whitespace from both sides.
        norm_v = re.sub(r"\s+", "", str(value)).casefold()
        norm_t = re.sub(r"\s+", "", ocr_text).casefold()
        return norm_v in norm_t
    if field_name == "invoice_date":
        try:
            d = date_parser.parse(str(value), dayfirst=True, fuzzy=True).date()
        except (ValueError, TypeError, OverflowError):
            return False
        # Try common date string forms — receipt may use any of these.
        forms = [
            d.isoformat(),
            d.strftime("%d/%m/%Y"),
            d.strftime("%d-%m-%Y"),
            d.strftime("%d/%m/%y"),
            d.strftime("%d-%m-%y"),
            d.strftime("%m/%d/%Y"),
            d.strftime("%Y/%m/%d"),
            d.strftime("%d %b %Y"),
            d.strftime("%d-%b-%Y"),
            d.strftime("%d %B %Y"),
        ]
        norm_text = _normalize_str(ocr_text)
        return any(_normalize_str(f) in norm_text for f in forms)
    if field_name == "total_amount":
        try:
            v = float(value)
        except (TypeError, ValueError):
            return False
        # Try whole-number, two-decimal, raw, and bare-int representations.
        forms = [f"{v:.2f}", f"{v:.1f}", str(v)]
        if v == int(v):
            forms.append(str(int(v)))
        # Strip thousands separators from text so "1,234.56" matches "1234.56".
        norm_text = _normalize_str(ocr_text).replace(",", "")
        return any(form in norm_text for form in forms)
    if field_name == "currency":
        code = str(value).strip().upper()
        tokens = CURRENCY_TOKENS.get(code, [code])
        norm_text = _normalize_str(ocr_text)
        return any(_normalize_str(t) in norm_text for t in tokens)
    # Fields outside the scalar set (e.g. line_items) — out of scope for v1.
    return True


# ---------- record-level evaluation ----------

@dataclass
class RecordResult:
    """Per-record scoring output. Aggregated by `aggregate()`."""

    invoice_id: str
    schema_valid: bool
    fields_evaluated: list[str]            # keys present in expected_json (intersected w/ COMPARATORS)
    field_matches: dict[str, bool]         # per-field correctness (only for evaluated fields)
    fields_predicted: list[str]            # scalar fields where prediction was non-null
    hallucinated_fields: list[str]         # subset of fields_predicted not findable in OCR text
    fully_correct: bool                    # every evaluated field matched (and at least 1 was evaluated)


def evaluate_record(
    *,
    invoice_id: str,
    prediction: dict[str, Any] | None,
    expected: dict[str, Any],
    ocr_text: str,
    raw_output: str | None = None,
) -> RecordResult:
    """Score a single record.

    Args:
        prediction: parsed model output as dict, or None if parsing failed.
        expected: sparse ground-truth dict (only fields we actually have).
        ocr_text: original OCR input — used for the hallucination check.
        raw_output: original API response string. Used to determine schema_valid
            when `prediction` is None (e.g., the model returned malformed JSON).
    """
    # Parse failure path
    if prediction is None:
        schema_valid = is_schema_valid(raw_output)
        evaluated = [k for k in expected if k in COMPARATORS]
        return RecordResult(
            invoice_id=invoice_id,
            schema_valid=schema_valid,
            fields_evaluated=evaluated,
            field_matches={k: False for k in evaluated},
            fields_predicted=[],
            hallucinated_fields=[],
            fully_correct=False,
        )

    schema_valid = is_schema_valid(prediction)

    # Field-level scoring — only the keys we have GT for.
    field_matches: dict[str, bool] = {}
    for key, exp_val in expected.items():
        if key not in COMPARATORS:
            continue  # e.g. line_items — skip in v1
        field_matches[key] = COMPARATORS[key](prediction.get(key), exp_val)

    # Hallucination check: every non-null scalar prediction must be findable in OCR.
    fields_predicted: list[str] = []
    hallucinated: list[str] = []
    for key in SCALAR_FIELDS:
        pred_val = prediction.get(key)
        if pred_val is None:
            continue
        fields_predicted.append(key)
        if not is_grounded(key, pred_val, ocr_text):
            hallucinated.append(key)

    fields_evaluated = list(field_matches.keys())
    fully_correct = bool(fields_evaluated) and all(field_matches.values())

    return RecordResult(
        invoice_id=invoice_id,
        schema_valid=schema_valid,
        fields_evaluated=fields_evaluated,
        field_matches=field_matches,
        fields_predicted=fields_predicted,
        hallucinated_fields=hallucinated,
        fully_correct=fully_correct,
    )


# ---------- aggregation ----------

@dataclass
class FieldStats:
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


@dataclass
class AggregateMetrics:
    """Predictor-level summary. Serializable via dataclasses.asdict for the report."""

    n_records: int
    schema_validity: float                          # records with schema-valid output / n_records
    field_accuracy: dict[str, FieldStats] = field(default_factory=dict)
    field_accuracy_macro: float = 0.0               # mean of per-field accuracies (table column)
    field_accuracy_micro: float = 0.0               # sum(correct) / sum(total) across fields
    record_accuracy: float = 0.0                    # records where every evaluated field matched
    hallucination_rate: float = 0.0                 # ungrounded predictions / total non-null predictions
    n_predictions: int = 0                          # total non-null scalar predictions
    n_hallucinations: int = 0                       # subset that weren't grounded


def aggregate(results: list[RecordResult]) -> AggregateMetrics:
    n = len(results)
    if n == 0:
        return AggregateMetrics(n_records=0, schema_validity=0.0)

    n_schema_valid = sum(1 for r in results if r.schema_valid)
    n_record_correct = sum(1 for r in results if r.fully_correct)

    field_stats: dict[str, FieldStats] = {}
    for r in results:
        for key, matched in r.field_matches.items():
            stats = field_stats.setdefault(key, FieldStats(correct=0, total=0))
            stats.total += 1
            if matched:
                stats.correct += 1

    if field_stats:
        macro = sum(s.accuracy for s in field_stats.values()) / len(field_stats)
        total_correct = sum(s.correct for s in field_stats.values())
        total_evaluated = sum(s.total for s in field_stats.values())
        micro = total_correct / total_evaluated if total_evaluated else 0.0
    else:
        macro = 0.0
        micro = 0.0

    n_predictions = sum(len(r.fields_predicted) for r in results)
    n_hallucinations = sum(len(r.hallucinated_fields) for r in results)
    halluc_rate = n_hallucinations / n_predictions if n_predictions else 0.0

    return AggregateMetrics(
        n_records=n,
        schema_validity=n_schema_valid / n,
        field_accuracy=field_stats,
        field_accuracy_macro=macro,
        field_accuracy_micro=micro,
        record_accuracy=n_record_correct / n,
        hallucination_rate=halluc_rate,
        n_predictions=n_predictions,
        n_hallucinations=n_hallucinations,
    )
