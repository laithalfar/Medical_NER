"""CLI utility to run the medspaCy + MeSH NER pipeline over input text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from data.preprocessing import PreprocessingPipeline
from medspacy_pipeline import EntityExtraction, MedspacyNERPipeline

# Function to parse command line arguments
def parse_args() -> argparse.Namespace:

    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Detect medical entities with medspaCy and link them to MeSH."
    )

    # Add arguments
    parser.add_argument("--text", help = "Raw text to analyze.")
    parser.add_argument(
        "--text-file",
        type = Path,
        help = "Path to a UTF-8 text file. Each line is treated as a separate document.",
    )

    # Add MeSH arguments
    parser.add_argument("--mesh-xml", type = Path, help = "Path to MeSH descriptor XML (desc*.xml).")
    parser.add_argument(
        "--mesh-umls-map",
        type=Path,
        help="Optional JSON/TSV file mapping MeSH IDs to UMLS CUIs.",
    )

    # Add MeSH loading arguments
    parser.add_argument(
        "--mesh-max-records",
        type=int,
        help="Limit number of MeSH descriptors to load (useful for prototyping).",
    )

    # Add MeSH term arguments
    parser.add_argument(
        "--mesh-include-term",
        dest="mesh_include_terms",
        action="append",
        help="Restrict MeSH loading to the provided preferred term. Repeatable.",
    )

    # Add preprocessing arguments
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip the cleaning/tokenization step and feed raw text to medspaCy.",
    )

    # Add output arguments
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit machine-readable JSON instead of pretty text output.",
    )

    # Parse arguments
    args = parser.parse_args()

    # Check if text or text-file is provided
    if not args.text and not args.text_file:
        parser.error("Provide --text or --text-file to analyze input.")
    return args

# Function to read inputs from command line arguments
def read_inputs(args: argparse.Namespace) -> List[str]:
    docs: List[str] = []

    # Check if text is provided
    if args.text:
        docs.append(args.text)
    
    # Check if text-file is provided and read it line by line
    if args.text_file:
        file_texts = [line.strip() for line in args.text_file.read_text(encoding="utf-8").splitlines()]
        docs.extend([txt for txt in file_texts if txt])
    return docs

# change entity to dictionary
def to_dict(entity: EntityExtraction, cleaned_text: str) -> dict:
    return {
        "text": entity.text,
        "label": entity.label,
        "start_char": entity.start_char,
        "end_char": entity.end_char,
        "mesh_id": entity.concept_id,
        "canonical_name": entity.canonical_name,
        "entity_type": entity.entity_type,
        "ontology": entity.ontology,
        "umls_cui": entity.umls_cui,
        "cleaned_text": cleaned_text,
    }

# format output to be pretty
def format_pretty(doc: str, cleaned_text: str, entities: Iterable[EntityExtraction]) -> str:
    lines = ["Input:", doc]
    if cleaned_text != doc:
        lines.extend(["Cleaned:", cleaned_text])
    lines.append("Entities:")
    for ent in entities:
        lines.append(
            f"- {ent.text!r} [{ent.label}] -> MeSH={ent.concept_id or 'N/A'}"
            f" ({ent.canonical_name or 'unknown'}, type={ent.entity_type or 'N/A'}, CUI={ent.umls_cui or 'N/A'})"
        )
    return "\n".join(lines)

# main function
def main() -> None:

    # parse arguments
    args = parse_args()
    docs = read_inputs(args)

    # create pipeline
    pipeline = MedspacyNERPipeline(
        mesh_descriptor_path=str(args.mesh_xml) if args.mesh_xml else None,
        mesh_to_umls_map=str(args.mesh_umls_map) if args.mesh_umls_map else None,
        mesh_max_records=args.mesh_max_records,
        mesh_include_terms=args.mesh_include_terms,
    )

    # create preprocessor
    preprocessor = None if args.no_preprocess else PreprocessingPipeline()

    # create outputs
    outputs = []
    for doc in docs:
        cleaned = doc if args.no_preprocess else preprocessor.clean_text(doc)
        entities = pipeline.analyze(cleaned)
        if args.as_json:
            outputs.append({"raw_text": doc, "entities": [to_dict(ent, cleaned) for ent in entities]})
        else:
            outputs.append(format_pretty(doc, cleaned, entities))

    if args.as_json:
        print(json.dumps(outputs, indent=2))
    else:
        print("\n\n".join(outputs))


if __name__ == "__main__":
    main()
