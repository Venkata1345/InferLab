"""System + user prompt templates for invoice extraction.

Kept minimal because we rely on schema-constrained decoding (vLLM guided_json
or OpenAI / Gemini structured output) for *format* enforcement. The prompt
covers *intent* — what to do for missing fields, how to normalize dates and
currency. Format constraints live in service/schema.py.
"""

SYSTEM_PROMPT = """\
You extract structured invoice data from OCR'd receipt text. Return only JSON conforming to the provided schema.

Rules:
- If a field is not present in the input, use null (or an empty list for line_items). Do not invent values.
- Dates: convert to ISO YYYY-MM-DD. Assume day-first order (DD/MM/YYYY) when ambiguous.
- Currency: ISO 4217 code (USD, MYR, EUR, GBP, ...). Map symbols: $ -> USD, RM -> MYR, EUR -> EUR, GBP -> GBP. Use null if currency cannot be determined.
- total_amount: the FINAL amount due after taxes and discounts. Strip currency symbols and parse as a number.
- line_items: each row of the itemized list. Use the amount as printed; do not recalculate."""


def build_user_message(ocr_text: str) -> str:
    """Wrap OCR text into the user-message body."""
    return f"Invoice text:\n\n{ocr_text}"


def build_messages(ocr_text: str) -> list[dict[str, str]]:
    """OpenAI-format chat messages (works for OpenAI API + vLLM OpenAI-compatible endpoint)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(ocr_text)},
    ]
