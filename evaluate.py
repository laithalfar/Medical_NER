"""Evaluate the medspaCy NER pipeline against BRAT-formatted annotations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from medspacy_pipeline import MedspacyNERPipeline

# variable class
@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str
    text: str


DocId = str
SpanKey = Tuple[DocId, int, int, str]

# Use a dict to match the gold span label with the predicted labels
DEFAULT_LABEL_MAP = {
    "DISEASE_DISORDER": "DISEASE",
    "DISEASE": "DISEASE",
    "SIGN_SYMPTOM": "SYMPTOM",
    "SYMPTOM": "SYMPTOM",
    "THERAPEUTIC_PROCEDURE": "THERAPEUTIC_PROCEDURE",
    "DIAGNOSTIC_PROCEDURE": "DIAGNOSTIC_PROCEDURE",
    "MEDICATION": "MEDICATION",
}


def parse_args() -> argparse.Namespace:

    # parse arguments
    parser = argparse.ArgumentParser(description="Evaluate medspaCy NER vs. BRAT annotations")
    parser.add_argument("dataset_dir", type=Path, help="Directory containing paired .txt/.ann files")
    parser.add_argument("--limit", type=int, help="Optional maximum number of documents to evaluate")
    parser.add_argument("--mesh-xml", type=Path, help="Path to MeSH descriptor XML (desc*.xml)")
    parser.add_argument("--mesh-umls-map", type=Path, help="Optional JSON/TSV mapping of MeSH IDs to UMLS CUIs")
    parser.add_argument("--mesh-max-records", type=int, help="Limit number of MeSH descriptors to load")
    parser.add_argument(
        "--mesh-include-term",
        dest="mesh_include_terms",
        action="append",
        help="Restrict the MeSH load to these preferred terms (repeatable)",
    )
    parser.add_argument(
        "--kb-json",
        dest="kb_json_paths",
        action="append",
        help="Additional KB JSON files to merge (e.g., maccrabat_kb.json)",
    )
    parser.add_argument(
        "--entity-type",
        dest="allowed_entity_types",
        action="append",
        help="Restrict the pipeline to these entity types (repeatable, e.g., DISEASE)",
    )
    parser.add_argument(
        "--target-label",
        dest="target_labels",
        action="append",
        help="Only score these labels (after mapping). Defaults to all labels present.",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        help="Optional JSON file mapping labels (gold -> evaluation space). Defaults to built-in mapping.",
    )
    parser.add_argument("--output-json", type=Path, help="Optional path to dump metrics as JSON")
    return parser.parse_args()

# iterate through documents
def iter_documents(dataset_dir: Path, limit: Optional[int] = None):
    count = 0
    
    # go through sorted .txt files in path
    for txt_path in sorted(dataset_dir.glob("*.txt")):
        
        # Search for .ann equal
        ann_path = txt_path.with_suffix(".ann")
        if not ann_path.exists():
            continue

        # Return stem of the text file(gives you the document ID (filename without extension)), 
        # the text inside the file and loads the brat(gold) annotations of ann_pth
        yield txt_path.stem, txt_path.read_text(encoding="utf-8"), load_brat_annotations(ann_path)

        # Perform task until all documents are done
        count += 1
        if limit and count >= limit:
            break

# load brat annotations
def load_brat_annotations(path: Path) -> List[Span]:
    spans: List[Span] = []

    # split and read text from files
    for line in path.read_text(encoding="utf-8").splitlines():

        # if line does not start with T skip line
        if not line or not line.startswith("T"):
            continue

        #else try splitting line into two parts
        try:
            _, meta, text = line.split("\t", 2)
        except ValueError:
            continue

        #Split meta
        parts = meta.split()

        #if meta splitsa into less than 3 parts skip it
        if len(parts) < 3:
            continue
        #else join all the parts in a variable and label in another variable
        label = parts[0]
        offset_str = " ".join(parts[1:])

        # go through string after it is split by ";"
        for block in offset_str.split(";"):
            block = block.strip()
            if not block:
                continue

            #split each block into start and end strings
            try:
                start_str, end_str = block.split()

                #store start and end as integers
                start, end = int(start_str), int(end_str)
            except ValueError:
                continue
            
            #store data as spans
            spans.append(Span(start=start, end=end, label=label.upper(), text=text))
    return spans

# builds the pipeline
def build_pipeline(args: argparse.Namespace, allowed_entity_types: Optional[Sequence[str]]) -> MedspacyNERPipeline:
    return MedspacyNERPipeline(
        mesh_descriptor_path=str(args.mesh_xml) if args.mesh_xml else None,
        mesh_to_umls_map=str(args.mesh_umls_map) if args.mesh_umls_map else None,
        mesh_max_records=args.mesh_max_records,
        mesh_include_terms=args.mesh_include_terms,
        allowed_entity_types=allowed_entity_types,
        kb_json_paths=args.kb_json_paths,
    )

# filters the spans based on target labels
def filter_labels(spans: Iterable[Span], target_labels: Optional[Set[str]]) -> List[Span]:
    if not target_labels:
        return list(spans)
    return [span for span in spans if span.label in target_labels]


# return span key with doc_id, start, end and label
def span_key(doc_id: str, span: Span) -> SpanKey:
    return (doc_id, span.start, span.end, span.label)

# change from entity to span
def entity_to_span(entity, doc_id: str) -> Span:
    label = (entity.entity_type or entity.label or "").upper()
    
    # span of an entity
    return Span(start=entity.start_char, end=entity.end_char, label=label, text=entity.text)

# remaps the labels based on the label map (matches gold span label with predicted label)
def remap_labels(spans: Iterable[Span], label_map: Dict[str, str]) -> List[Span]:
    remapped: List[Span] = []
    for span in spans:
        mapped_label = label_map.get(span.label.upper(), span.label.upper())
        remapped.append(Span(start=span.start, end=span.end, label=mapped_label, text=span.text))
    return remapped

# loads the label map as a dictionary
def load_label_map(path: Optional[Path]) -> Dict[str, str]:
    
    # if no path is provided, use the default label map
    # load its key(k) and value(v)
    if not path:
        return {k.upper(): v.upper() for k, v in DEFAULT_LABEL_MAP.items()}
    
    # else load the label map from the path 
    # load its key(k) and value(v)
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k).upper(): str(v).upper() for k, v in data.items()}

# compute metrics
def compute_metrics(pred_map: Dict[SpanKey, Span], gold_map: Dict[SpanKey, Span]) -> Dict[str, object]:
    pred_keys = set(pred_map.keys())
    gold_keys = set(gold_map.keys())
    tp_keys = pred_keys & gold_keys
    fp_keys = pred_keys - gold_keys
    fn_keys = gold_keys - pred_keys

    # avoid dividing by 0
    def safe_div(num: int, denom: int) -> float:
        return num / denom if denom else 0.0

    #instantiate metric variables
    overall = {
        "tp": len(tp_keys),
        "fp": len(fp_keys),
        "fn": len(fn_keys),
        "precision": safe_div(len(tp_keys), len(pred_keys)),
        "recall": safe_div(len(tp_keys), len(gold_keys)),
    }

    #calculate f1
    overall["f1"] = (
        safe_div(2 * overall["precision"] * overall["recall"], overall["precision"] + overall["recall"])
        if (overall["precision"] + overall["recall"]) > 0
        else 0.0
    )

    # Sort how many of each tp, fp, and fn there is
    per_label = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for key in tp_keys:
        per_label[key[3]]["tp"] += 1
    for key in fp_keys:
        per_label[key[3]]["fp"] += 1
    for key in fn_keys:
        per_label[key[3]]["fn"] += 1

    # do calculations for metrics
    for label, stats in per_label.items():
        stats["precision"] = safe_div(stats["tp"], stats["tp"] + stats["fp"])
        stats["recall"] = safe_div(stats["tp"], stats["tp"] + stats["fn"])
        denom = stats["precision"] + stats["recall"]
        stats["f1"] = safe_div(2 * stats["precision"] * stats["recall"], denom) if denom else 0.0

    return {
        "overall": overall,
        "per_label": dict(per_label),
        "false_positives": [pred_map[key] for key in fp_keys],
        "false_negatives": [gold_map[key] for key in fn_keys],
    }


def main() -> None:
    
    # parse arguments
    args = parse_args()
    label_map = load_label_map(args.label_map)

    # determine allowed entity types
    allowed_entity_types = (
        # if allowed entity types are provided from parsing, remap them
        [label_map.get(label.upper(), label.upper()) for label in args.allowed_entity_types]
        if args.allowed_entity_types
        else None
    )

    # determine target labels
    target_labels = (
        # if target labels are provided from parsing, remap them
        {label_map.get(label.upper(), label.upper()) for label in args.target_labels}
        if args.target_labels
        else None
    )

    # load documents
    docs = list(iter_documents(args.dataset_dir, limit=args.limit))
    if not docs:
        raise SystemExit(f"No annotated documents found under {args.dataset_dir}")

    # build pipeline
    pipeline = build_pipeline(args, allowed_entity_types)

    # pred and gold map
    pred_map: Dict[SpanKey, Span] = {}
    gold_map: Dict[SpanKey, Span] = {}

    # iterate through documents id, text, gold spans    
    for doc_id, text, gold_spans in docs:
        # remap and filter gold spans
        gold_mapped = remap_labels(gold_spans, label_map)
        gold_filtered = filter_labels(gold_mapped, target_labels)

        #store all gold spans in a dict
        gold_map.update({span_key(doc_id, span): span for span in gold_filtered})

        # get, remap and filter predicted spans
        predicted = pipeline.analyze(text)
        pred_spans = remap_labels([entity_to_span(ent, doc_id) for ent in predicted], label_map)

        # filter predicted spans
        if target_labels:
            pred_spans = [span for span in pred_spans if span.label in target_labels]

        #store all predicted spans in a dict
        pred_map.update({span_key(doc_id, span): span for span in pred_spans})

    # compute metrics
    metrics = compute_metrics(pred_map, gold_map)
    overall = metrics["overall"]

    # Print results
    print("Documents evaluated:", len(docs))
    print("Gold entities:", overall["tp"] + overall["fn"])
    print("Predicted entities:", overall["tp"] + overall["fp"])
    print(
        "Precision: {:.3f} | Recall: {:.3f} | F1: {:.3f}".format(
            overall["precision"], overall["recall"], overall["f1"]
        )
    )

    print("\nPer-label metrics:")
    for label, stats in sorted(metrics["per_label"].items()):
        print(
            f"  {label:<18} P={stats['precision']:.3f} R={stats['recall']:.3f} F1={stats['f1']:.3f}"
            f" (TP={stats['tp']} FP={stats['fp']} FN={stats['fn']})"
        )

    # print up to the first 5 samples    
    def sample(spans: List[Span], title: str) -> None:
        print(f"\nSample {title} (up to 5):")
        for span in spans[:5]:
            print(f"  • [{span.label}] '{span.text}' ({span.start}-{span.end})")

    # print first 5 from false_positives
    sample(metrics["false_positives"], "false positives")
    sample(metrics["false_negatives"], "false negatives")

    #save metrics to output_json
    if args.output_json:
        args.output_json.write_text(json.dumps(metrics, default=_span_to_dict, indent=2), encoding="utf-8")
        print(f"\nDetailed metrics saved to {args.output_json}")

# helper function to convert span to dict
def _span_to_dict(span: Span) -> Dict[str, object]:  # pragma: no cover
    return {"start": span.start, "end": span.end, "label": span.label, "text": span.text}


if __name__ == "__main__":
    main()
