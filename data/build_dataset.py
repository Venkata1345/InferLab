"""Download SROIE, map fields to InferLab schema, write train/eval JSONL.

Source: github.com/zzzDavid/ICDAR-2019-SROIE @ pinned commit (no auth required).
We pull only data/key/*.json (ground truth) + data/box/*.csv (OCR text + bbox);
images aren't needed since we treat OCR as already done.

Sparse mapping (only fields with ground truth — never faked):
    SROIE company → vendor_name
    SROIE date    → invoice_date  (normalized to YYYY-MM-DD, dayfirst)
    SROIE total   → total_amount  (parsed to float, currency stripped)
SROIE has no invoice_number / currency / line_items → omitted from expected_json.
The eval harness scores only the keys present in expected_json.

Run: python -m data.build_dataset
"""

import argparse
import json
import random
import re
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dateutil import parser as date_parser
from tqdm import tqdm

SROIE_COMMIT = "27be4271b251c256f695acbade9a801bffe85994"
SROIE_TARBALL_URL = f"https://github.com/zzzDavid/ICDAR-2019-SROIE/archive/{SROIE_COMMIT}.tar.gz"
EXPECTED_RECORD_COUNT = 626

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

DEFAULT_SEED = 42
DEFAULT_EVAL_FRAC = 0.2


@dataclass(frozen=True)
class SroieRecord:
    """One SROIE invoice: invoice id + OCR text + raw ground-truth dict."""

    invoice_id: str
    ocr_text: str
    raw_key: dict[str, str]


# ---------- download ----------


def download_sroie(raw_dir: Path) -> None:
    """Download + extract data/key/*.json and data/box/*.csv from the pinned mirror.

    Idempotent: skips when both directories already contain EXPECTED_RECORD_COUNT files.
    """
    key_dir = raw_dir / "key"
    box_dir = raw_dir / "box"
    if _already_downloaded(key_dir, box_dir):
        print(f"SROIE already present in {raw_dir} -- skipping download.", file=sys.stderr)
        return

    raw_dir.mkdir(parents=True, exist_ok=True)
    tarball_path = raw_dir / f"sroie-{SROIE_COMMIT[:8]}.tar.gz"

    if not tarball_path.exists():
        print(f"Downloading SROIE tarball from {SROIE_TARBALL_URL}", file=sys.stderr)
        _stream_download(SROIE_TARBALL_URL, tarball_path)

    print("Extracting data/key/*.json + data/box/*.csv ...", file=sys.stderr)
    key_dir.mkdir(parents=True, exist_ok=True)
    box_dir.mkdir(parents=True, exist_ok=True)
    _extract_key_box(tarball_path, key_dir, box_dir)

    n_key = len(list(key_dir.glob("*.json")))
    n_box = len(list(box_dir.glob("*.csv")))
    if n_key != EXPECTED_RECORD_COUNT or n_box != EXPECTED_RECORD_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_RECORD_COUNT} key + {EXPECTED_RECORD_COUNT} box files, "
            f"got {n_key} key + {n_box} box. Tarball may be incomplete."
        )


def _stream_download(url: str, dest: Path) -> None:
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        with (
            dest.open("wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar,
        ):
            for chunk in r.iter_bytes(chunk_size=64 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))


def _extract_key_box(tarball_path: Path, key_dir: Path, box_dir: Path) -> None:
    with tarfile.open(tarball_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            parts = member.name.split("/", 1)
            if len(parts) < 2:
                continue
            rel = parts[1]
            if rel.startswith("data/key/") and rel.endswith(".json"):
                _safe_write(tar, member, key_dir, rel[len("data/key/") :])
            elif rel.startswith("data/box/") and rel.endswith(".csv"):
                _safe_write(tar, member, box_dir, rel[len("data/box/") :])


def _safe_write(
    tar: tarfile.TarFile, member: tarfile.TarInfo, dest_dir: Path, basename: str
) -> None:
    if "/" in basename or basename.startswith("..") or not basename:
        return
    src = tar.extractfile(member)
    if src is None:
        return
    (dest_dir / basename).write_bytes(src.read())


def _already_downloaded(key_dir: Path, box_dir: Path) -> bool:
    if not key_dir.is_dir() or not box_dir.is_dir():
        return False
    return (
        len(list(key_dir.glob("*.json"))) == EXPECTED_RECORD_COUNT
        and len(list(box_dir.glob("*.csv"))) == EXPECTED_RECORD_COUNT
    )


# ---------- load + parse ----------


def load_records(raw_dir: Path) -> list[SroieRecord]:
    """Pair each key/NNN.json with box/NNN.csv → SroieRecord list, sorted by id."""
    key_dir = raw_dir / "key"
    box_dir = raw_dir / "box"
    records: list[SroieRecord] = []
    for key_path in sorted(key_dir.glob("*.json")):
        invoice_id = key_path.stem
        box_path = box_dir / f"{invoice_id}.csv"
        if not box_path.exists():
            continue
        try:
            raw_key = json.loads(key_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        ocr_text = parse_box_csv(box_path)
        records.append(SroieRecord(invoice_id=invoice_id, ocr_text=ocr_text, raw_key=raw_key))
    return records


def parse_box_csv(path: Path) -> str:
    """SROIE box CSV → joined OCR text in document (top-down) order.

    Per-line format: x1,y1,x2,y2,x3,y3,x4,y4,text  — text may contain commas,
    so we split on the first 8 commas and take everything after as the text token.
    """
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.strip():
            continue
        parts = raw.split(",", 8)
        if len(parts) < 9:
            continue
        text = parts[8].strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


# ---------- field normalization ----------

# Strip everything except digits, dot, minus — handles "RM 9.00", "$1,234.56", etc.
_TOTAL_STRIP_RE = re.compile(r"[^\d.\-]")


def normalize_total(raw: str | None) -> float | None:
    if not raw:
        return None
    s = raw.strip().replace(",", "")
    s = _TOTAL_STRIP_RE.sub("", s)
    if not s or s in {".", "-", "-."}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def normalize_date(raw: str | None) -> str | None:
    """SROIE dates → ISO YYYY-MM-DD. Malaysian receipts → dayfirst=True."""
    if not raw:
        return None
    s = raw.strip()
    if not s:
        return None
    try:
        dt = date_parser.parse(s, dayfirst=True, fuzzy=True)
    except (ValueError, TypeError, OverflowError):
        return None
    return dt.date().isoformat()


def map_to_schema(raw_key: dict[str, Any]) -> dict[str, Any]:
    """SROIE ground truth → InferLab sparse expected_json (only fields we actually know)."""
    out: dict[str, Any] = {}
    company = (raw_key.get("company") or "").strip()
    if company:
        out["vendor_name"] = company
    date_iso = normalize_date(raw_key.get("date"))
    if date_iso is not None:
        out["invoice_date"] = date_iso
    total = normalize_total(raw_key.get("total"))
    if total is not None:
        out["total_amount"] = total
    # invoice_number / currency / line_items: not in SROIE → omit (sparse)
    return out


# ---------- build split ----------


def build_split(
    raw_dir: Path,
    processed_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    eval_frac: float = DEFAULT_EVAL_FRAC,
) -> tuple[int, int, list[SroieRecord]]:
    """Build train.jsonl + eval.jsonl. Returns (n_train, n_eval, eval_records)."""
    records = load_records(raw_dir)
    if not records:
        raise RuntimeError(f"No SROIE records under {raw_dir}. Run download_sroie() first.")

    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_eval = max(1, int(round(len(shuffled) * eval_frac)))
    eval_records = shuffled[:n_eval]
    train_records = shuffled[n_eval:]

    processed_dir.mkdir(parents=True, exist_ok=True)
    n_train_written = _write_jsonl(processed_dir / "train.jsonl", train_records)
    n_eval_written = _write_jsonl(processed_dir / "eval.jsonl", eval_records)
    return n_train_written, n_eval_written, eval_records


def _write_jsonl(path: Path, records: list[SroieRecord]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            expected = map_to_schema(rec.raw_key)
            if not expected:
                continue  # all fields unparseable — skip rather than write empty truth
            row = {
                "invoice_id": rec.invoice_id,
                "input_text": rec.ocr_text,
                "expected_json": expected,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


# ---------- verify ----------


def verify_jsonl(path: Path) -> tuple[int, list[str]]:
    """Re-read a JSONL and check every row parses + has at least one expected field.

    Returns (n_ok, problems). problems is empty when every row is well-formed.
    """
    n_ok = 0
    problems: list[str] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                problems.append(f"{path.name}:{lineno} not valid JSON ({e})")
                continue
            for required in ("invoice_id", "input_text", "expected_json"):
                if required not in row:
                    problems.append(f"{path.name}:{lineno} missing key {required!r}")
                    break
            else:
                expected = row["expected_json"]
                if not isinstance(expected, dict) or not expected:
                    problems.append(f"{path.name}:{lineno} expected_json empty/non-dict")
                    continue
                # Light schema checks on the keys we do produce
                if "invoice_date" in expected and not _looks_like_iso_date(
                    expected["invoice_date"]
                ):
                    problems.append(
                        f"{path.name}:{lineno} invoice_date {expected['invoice_date']!r} not ISO"
                    )
                    continue
                if "total_amount" in expected and not isinstance(
                    expected["total_amount"], (int, float)
                ):
                    problems.append(
                        f"{path.name}:{lineno} total_amount {expected['total_amount']!r} not numeric"
                    )
                    continue
                n_ok += 1
    return n_ok, problems


_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _looks_like_iso_date(s: object) -> bool:
    return isinstance(s, str) and bool(_ISO_DATE_RE.match(s))


# ---------- sample preview ----------


def preview_samples(eval_records: list[SroieRecord], n: int = 3, seed: int = DEFAULT_SEED) -> None:
    """Print n records side-by-side: SROIE raw fields → mapped expected_json."""
    rng = random.Random(seed + 1)  # different from split seed so we get a varied preview
    picks = rng.sample(eval_records, min(n, len(eval_records)))
    for i, rec in enumerate(picks, 1):
        mapped = map_to_schema(rec.raw_key)
        print(f"\n=== Sample {i} (invoice {rec.invoice_id}) ===")
        print("SROIE raw:")
        for k in ("company", "date", "address", "total"):
            v = rec.raw_key.get(k, "<missing>")
            print(f"  {k:<8}: {v}")
        print("Mapped expected_json:")
        for line in json.dumps(mapped, indent=2, ensure_ascii=False).splitlines():
            print(f"  {line}")
        snippet = rec.ocr_text[:200].replace("\n", " | ")
        print(f"OCR text (first 200 chars):\n  {snippet}")


# ---------- main ----------


def main() -> None:
    # Windows default cp1252 chokes on non-ASCII OCR text — force UTF-8 output.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--eval-frac", type=float, default=DEFAULT_EVAL_FRAC)
    ap.add_argument("--samples", type=int, default=3, help="How many sample records to preview")
    args = ap.parse_args()

    download_sroie(args.raw_dir)
    n_train, n_eval, eval_records = build_split(
        args.raw_dir, args.processed_dir, seed=args.seed, eval_frac=args.eval_frac
    )
    train_path = args.processed_dir / "train.jsonl"
    eval_path = args.processed_dir / "eval.jsonl"
    print(f"\nWrote {n_train} train records -> {train_path}")
    print(f"Wrote {n_eval} eval records  -> {eval_path}")

    if args.samples:
        preview_samples(eval_records, n=args.samples, seed=args.seed)

    print("\nVerifying eval.jsonl...")
    n_ok, problems = verify_jsonl(eval_path)
    if problems:
        print(f"  {len(problems)} problem(s):", file=sys.stderr)
        for p in problems[:10]:
            print(f"    - {p}", file=sys.stderr)
        if len(problems) > 10:
            print(f"    ... and {len(problems) - 10} more", file=sys.stderr)
        raise SystemExit(1)
    print(f"  OK: {n_ok}/{n_eval} eval records well-formed.")


if __name__ == "__main__":
    main()
