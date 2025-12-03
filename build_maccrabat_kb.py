"""Uses Maccrabat training data to extract needed data and store it as a dictionary in a json file for future training"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

# assign labels
DEFAULT_LABELS = (
    "DISEASE_DISORDER",
    "SIGN_SYMPTOM",
    "MEDICATION",
    "THERAPEUTIC_PROCEDURE",
    "DIAGNOSTIC_PROCEDURE",
)

# make a parse argument list for this as well
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MACCROBAT KB entries from .ann files")
    parser.add_argument("dataset_dir", type=Path, help="Directory containing MACCROBAT *.ann files")
    parser.add_argument("output_json", type=Path, help="Destination JSON file for KB entries")
    parser.add_argument(
        "--include-label",
        dest="include_labels",
        action="append",
        help="Restrict processing to these labels (repeatable). Defaults to a curated shortlist.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Only keep spans that appear at least this many times (default: 2).",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        help="Optional cap on the total number of KB entries (after filtering).",
    )
    return parser.parse_args()

# loading annotations
def load_annotations(dataset_dir: Path, include_labels: Sequence[str]) -> Dict[str, Counter]:
    
    # Get upper_case labels
    label_set = {label.upper() for label in include_labels}
    counts: Dict[str, Counter] = defaultdict(Counter)
    
    # Sort annotations 
    ann_files = sorted(dataset_dir.glob("*.ann"))
    if not ann_files:
        raise SystemExit(f"No .ann files found under {dataset_dir}")

    # Split paths from files into lines
    for ann_path in ann_files:
        for line in ann_path.read_text(encoding="utf-8").splitlines():
            if not line or not line.startswith("T"):
                continue
            try:
                _, meta, text = line.split("\t", 2)
            except ValueError:
                continue

            #split meta into parts
            parts = meta.split()
            if len(parts) < 3:
                continue

            # Take the first word from the split as the label
            label = parts[0].upper()
            if label_set and label not in label_set:
                continue
            cleaned = _clean_text(text)
            if not cleaned:
                continue
            #count how many repetition of this test and label there is and return the list of counts
            counts[label][cleaned] += 1
    return counts


# build entries from dictionaries
def build_entries(counts: Dict[str, Counter], min_count: int, max_entries: int | None) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []

    # get label, counter, text and frequency
    for label, counter in counts.items():
        for text, freq in counter.most_common():

            if freq < min_count:
                continue

            # Create concept_id from label and text
            concept_id = _make_concept_id(label, text)

            # get synonyms from text
            synonyms = list({text, text.lower()}) if text.lower() != text else [text]
            # create entry
            entries.append(
                {
                    "concept_id": concept_id,
                    "canonical_name": text,
                    "entity_type": label,
                    "synonyms": synonyms,
                    "description": f"Auto-generated from MACCROBAT ({freq} mentions)",
                    "ontology": "MACCROBAT",
                }
            )
            if max_entries and len(entries) >= max_entries:
                return entries
    return entries

# clean text by stripping and splitting it
def _clean_text(text: str) -> str:
    return " ".join(text.strip().split())

# create concept id using sha1 hashing
def _make_concept_id(label: str, text: str) -> str:
    digest = hashlib.sha1(f"{label}:{text}".encode("utf-8")).hexdigest()[:16]
    return f"MACCROBAT:{label}:{digest}"

# main function
def main() -> None:
    # parse arguments
    args = parse_args()
    include_labels = args.include_labels or DEFAULT_LABELS

    # load annotations
    counts = load_annotations(args.dataset_dir, include_labels)
    entries = build_entries(counts, min_count=args.min_count, max_entries=args.max_entries)

    # save entries to json
    args.output_json.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(entries)} KB entries to {args.output_json}")


if __name__ == "__main__":
    main()
