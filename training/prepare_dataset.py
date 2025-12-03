"""Convert BRAT-style MACCROBAT annotations into spaCy DocBin files for NER training."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import spacy
from spacy.tokens import Doc, DocBin


@dataclass
class SpanAnnotation:
    start: int
    end: int
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare spaCy training data from BRAT annotations")
    parser.add_argument("dataset_dir", type=Path, help="Directory containing paired .txt/.ann files")
    parser.add_argument("output_dir", type=Path, help="Directory to write train/dev/test .spacy files")
    parser.add_argument("--label-map", type=Path, help="Optional JSON file mapping labels to new names")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Fraction of docs for training (default 0.8)")
    parser.add_argument("--dev-ratio", type=float, default=0.1, help="Fraction for dev/validation (default 0.1)")
    parser.add_argument(
        "--seed", type=int, default=13, help="Random seed for shuffling when splitting the dataset"
    )
    parser.add_argument(
        "--limit", type=int, help="Optional limit on the number of documents to process (useful for smoke tests)"
    )
    return parser.parse_args()


def load_label_map(path: Optional[Path]) -> Dict[str, str]:
    if not path:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k.upper(): v.upper() for k, v in data.items()}


def iter_examples(dataset_dir: Path, limit: Optional[int]) -> Iterable[Tuple[str, str, List[SpanAnnotation]]]:
    count = 0
    for txt_path in sorted(dataset_dir.glob("*.txt")):
        ann_path = txt_path.with_suffix(".ann")
        if not ann_path.exists():
            continue
        text = txt_path.read_text(encoding="utf-8")
        spans = load_brat_annotations(ann_path)
        yield txt_path.stem, text, spans
        count += 1
        if limit and count >= limit:
            break


def load_brat_annotations(path: Path) -> List[SpanAnnotation]:
    spans: List[SpanAnnotation] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or not line.startswith("T"):
            continue
        try:
            _, meta, _ = line.split("\t", 2)
        except ValueError:
            continue
        parts = meta.split()
        if len(parts) < 3:
            continue
        label = parts[0].upper()
        offsets = " ".join(parts[1:]).split(";")
        for block in offsets:
            block = block.strip()
            if not block:
                continue
            try:
                start_str, end_str = block.split()
                start, end = int(start_str), int(end_str)
            except ValueError:
                continue
            spans.append(SpanAnnotation(start=start, end=end, label=label))
    return spans


def apply_label_map(spans: Iterable[SpanAnnotation], label_map: Dict[str, str]) -> List[SpanAnnotation]:
    remapped: List[SpanAnnotation] = []
    for span in spans:
        mapped = label_map.get(span.label, span.label)
        remapped.append(SpanAnnotation(start=span.start, end=span.end, label=mapped))
    return remapped


def make_doc(text: str, spans: Iterable[SpanAnnotation], nlp) -> Optional[Doc]:
    doc = nlp.make_doc(text)
    ents = []
    seen = set()
    for span in spans:
        key = (span.start, span.end)
        if key in seen:
            continue
        seen.add(key)
        span_obj = doc.char_span(span.start, span.end, label=span.label, alignment_mode="contract")
        if span_obj is None:
            continue
        ents.append(span_obj)
    if not ents:
        doc.ents = ()
    else:
        doc.ents = tuple(filter(None, ents))
    return doc


def split_data(items: List[Tuple[str, Doc]], train_ratio: float, dev_ratio: float) -> Dict[str, List[Doc]]:
    total = len(items)
    train_end = int(total * train_ratio)
    dev_end = train_end + int(total * dev_ratio)
    return {
        "train": [doc for _, doc in items[:train_end]],
        "dev": [doc for _, doc in items[train_end:dev_end]],
        "test": [doc for _, doc in items[dev_end:]],
    }


def save_docbins(split_docs: Dict[str, List[Doc]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split_name, docs in split_docs.items():
        docbin = DocBin(store_user_data=True)
        label_counts: Dict[str, int] = {}
        for doc in docs:
            docbin.add(doc)
            for ent in doc.ents:
                label_counts[ent.label_] = label_counts.get(ent.label_, 0) + 1
        path = output_dir / f"{split_name}.spacy"
        docbin.to_disk(path)
        summary[split_name] = {"docs": len(docs), "labels": label_counts}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.train_ratio + args.dev_ratio >= 1.0:
        raise SystemExit("train_ratio + dev_ratio must be < 1.0 to leave room for the test split")

    label_map = load_label_map(args.label_map)
    nlp = spacy.blank("en")

    items: List[Tuple[str, Doc]] = []
    for doc_id, text, spans in iter_examples(args.dataset_dir, args.limit):
        remapped_spans = apply_label_map(spans, label_map)
        doc = make_doc(text, remapped_spans, nlp)
        if doc is None:
            continue
        items.append((doc_id, doc))

    if not items:
        raise SystemExit("No documents were processed successfully.")

    random.Random(args.seed).shuffle(items)
    split_docs = split_data(items, args.train_ratio, args.dev_ratio)
    save_docbins(split_docs, args.output_dir)
    print(
        f"Wrote {len(split_docs['train'])} train, {len(split_docs['dev'])} dev,"
        f" and {len(split_docs['test'])} test docs to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
