import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from openai import APIError, APITimeoutError, OpenAI, RateLimitError


DEFAULT_INPUT = "관리자페이지 (씬그래프 리더스터디)의 사본 (최종_가영) - ALL.csv"
DEFAULT_MODEL = "gpt-5-2025-08-07"
TARGET_COLUMN_NAME = "Annotation_En"
DEFAULT_BATCH_SIZE = 5
DEFAULT_MAX_WORKERS = 16

SYSTEM_PROMPT = """You are a professional medical translator.
Translate the user's Korean annotations into English exactly and faithfully.

Rules:
- Do not add, omit, soften, or reinterpret content.
- Preserve uncertainty, hedging, severity, and clinical nuance.
- Keep medical terminology precise and natural in English.
- Do not summarize or reformat.
- Return only a JSON object.
- The output JSON must have exactly one key: "translations".
- "translations" must be a list of translated strings in the exact same order as the inputs.
- The number of output translations must exactly match the number of input items.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add Annotation_En next to annotation and fill it using the OpenAI GPT API."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        default="translated_all.csv",
        help="Output CSV path. If omitted, the input file is overwritten.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key. Defaults to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Translate only the first N eligible rows.",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=1,
        help="1-based data row index to start from.",
    )
    parser.add_argument(
        "--overwrite-filled",
        action="store_true",
        help="Re-translate rows even if Annotation_En already has a value.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.1,
        help="Delay between API requests in seconds.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of annotations to translate per API call.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Maximum number of batch requests to run in parallel.",
    )
    return parser.parse_args()


def normalize_header(value: str) -> str:
    return value.strip().lower()


def find_annotation_column(fieldnames: Iterable[str]) -> str:
    for name in fieldnames:
        if normalize_header(name) == "annotation":
            return name
    raise ValueError("Could not find an 'annotation' column in the CSV header.")


def build_fieldnames(original_fieldnames: list[str], annotation_column: str) -> list[str]:
    if TARGET_COLUMN_NAME in original_fieldnames:
        return original_fieldnames

    insert_at = original_fieldnames.index(annotation_column) + 1
    return (
        original_fieldnames[:insert_at]
        + [TARGET_COLUMN_NAME]
        + original_fieldnames[insert_at:]
    )


def build_batch_user_prompt(texts: list[str]) -> str:
    payload = [{"index": idx, "text": text} for idx, text in enumerate(texts)]
    return (
        "Translate each Korean annotation to English with no additions or omissions.\n"
        "Return JSON only.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def translate_batch(client: OpenAI, model: str, texts: list[str]) -> list[str]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_batch_user_prompt(texts)},
        ],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    parsed = json.loads(content)
    translations = parsed.get("translations")

    if not isinstance(translations, list):
        raise ValueError("Model response is missing a valid 'translations' list.")
    if len(translations) != len(texts):
        raise ValueError(
            f"Expected {len(texts)} translations, but received {len(translations)}."
        )

    cleaned: list[str] = []
    for item in translations:
        if not isinstance(item, str):
            raise ValueError("Each translated item must be a string.")
        cleaned.append(item.strip())
    return cleaned


def main() -> int:
    args = parse_args()

    if not args.api_key:
        print("OPENAI_API_KEY is required. Pass --api-key or set the environment variable.", file=sys.stderr)
        return 1

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("The CSV file is empty or missing a header row.")

        annotation_column = find_annotation_column(reader.fieldnames)
        fieldnames = build_fieldnames(list(reader.fieldnames), annotation_column)
        rows = list(reader)

    client = OpenAI(api_key=args.api_key)

    translated_count = 0
    eligible_seen = 0

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1.")

    def process_batch(batch: list[tuple[int, dict, str]]) -> list[tuple[int, dict, str]]:
        texts = [item[2] for item in batch]
        batch_rows = [str(item[0]) for item in batch]

        for attempt in range(5):
            try:
                translations = translate_batch(client, args.model, texts)
                if args.sleep > 0:
                    time.sleep(args.sleep)
                return [
                    (row_index, row, translation)
                    for (row_index, row, _source_text), translation in zip(batch, translations)
                ]
            except (RateLimitError, APITimeoutError, APIError, ValueError) as exc:
                if attempt == 4:
                    raise
                wait_seconds = min(2 ** attempt, 8)
                print(
                    f"[rows {', '.join(batch_rows)}] retrying after {wait_seconds}s because of an API error: {exc}",
                    file=sys.stderr,
                )
                time.sleep(wait_seconds)
        return []

    pending_batch: list[tuple[int, dict, str]] = []
    batches: list[list[tuple[int, dict, str]]] = []
    for row_index, row in enumerate(rows, start=1):
        source_text = (row.get(annotation_column) or "").strip()
        current_translation = (row.get(TARGET_COLUMN_NAME) or "").strip()

        if TARGET_COLUMN_NAME not in row:
            row[TARGET_COLUMN_NAME] = ""

        if row_index < args.start_row:
            continue
        if not source_text:
            continue
        if current_translation and not args.overwrite_filled:
            continue

        eligible_seen += 1
        if args.limit is not None and eligible_seen > args.limit:
            break

        pending_batch.append((row_index, row, source_text))
        if len(pending_batch) >= args.batch_size:
            batches.append(pending_batch)
            pending_batch = []

    if pending_batch:
        batches.append(pending_batch)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_batch = {
            executor.submit(process_batch, batch): batch
            for batch in batches
        }
        for future in as_completed(future_to_batch):
            results = future.result()
            for row_index, row, translation in results:
                row[TARGET_COLUMN_NAME] = translation
                translated_count += 1
                print(f"[row {row_index}] translated")

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output_path}")
    print(f"Rows translated: {translated_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
