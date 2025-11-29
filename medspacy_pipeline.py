"""medspaCy-based utilities for medical NER and ontology linking.

This module wires medspaCy's pre-built pipeline with a lightweight knowledge
base abstraction so we can quickly:

1. Detect medical entities (diseases, symptoms, medications, etc.).
2. Link the detected mentions to canonical concepts (e.g., MeSH IDs).

The knowledge base implementation is intentionally simple—loadable from JSON or
constructed from in-memory data—so it can be swapped with richer databases or
FHIR terminology services later on.
"""

from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import medspacy
from spacy.language import Language
from medspacy.ner import TargetMatcher, TargetRule

LOGGER = logging.getLogger(__name__) # log file name

# Regular expression to normalize aliases
_ALIAS_NORMALIZER = re.compile(r"[^a-z0-9]+")

# Use regular expression normalizer to replace characters with spaces after converting text to lowercase
def _normalize(text: str) -> str:
    return _ALIAS_NORMALIZER.sub(" ", text.lower()).strip()

# Create a class for KnowledgeBaseEntry for representing any canonical medical concept
# @dataclass is a decorator that is used to create a class with automatic building of
# constructor functions and other methods.
@dataclass
class KBEntry:
    """Representation of a canonical medical concept (MeSH by default)."""

    # instantiate variables
    concept_id: str
    canonical_name: str
    entity_type: str
    synonyms: List[str] = field(default_factory=list)
    description: Optional[str] = None
    cui: Optional[str] = None  # Optional UMLS mapping


    # Initiate variable ontology to "MeSH"
    ontology: str = "MeSH"

    # Return a list of aliases for the entry
    def aliases(self) -> List[str]:
        return [self.canonical_name, *self.synonyms]


# Create a class for MedicalKnowledgeBase
# @dataclass is a decorator that is used to create a class with automatic building of
# constructor functions and other methods.
@dataclass
class MedicalKnowledgeBase:
    """Very small in-memory knowledge base with alias matching."""

    # instantiate variables
    entries: Dict[str, KBEntry] = field(default_factory=dict)
    alias_index: Dict[str, str] = field(default_factory=dict)

    # Class method to create a MedicalKnowledgeBase from a JSON file
    @classmethod
    def from_json(cls, path: str | Path) -> "MedicalKnowledgeBase":
        data = json.loads(Path(path).read_text())
        kb = cls() # cls is the constructor function, it will construct class A and call the __init__(self, uid=None) function.
        for record in data:
            kb.add_entry(KBEntry(**record)) # add_entry is a method that adds a new entry to the knowledge 
            # base after it has been constructed.
        return kb

    # Class method to create a MedicalKnowledgeBase from a list of KBEntries
    @classmethod
    def from_entries(cls, entries: Iterable[KBEntry | Dict[str, object]]) -> "MedicalKnowledgeBase": 
        # entries could be of type KBEntry representing the output from function from_json or 
        # entries could be of type Dict[str, object] representing the input from variable entries (keep in mind KBEntry is an object as well)
        kb = cls()

        # for loop to check if entry is of type KBEntry or Dict[str, object]
        # and add it to KBclass accordingly
        for entry in entries:
            if isinstance(entry, KBEntry):
                kb.add_entry(entry)
            else:
                kb.add_entry(KBEntry(**entry))
        return kb

    # Class method to create a MedicalKnowledgeBase with demo entries
    @classmethod
    def with_demo_entries(cls) -> "MedicalKnowledgeBase":
        # call constructor to create a demo knowledge base
        # that we can fill with real data after
        return cls.from_entries(_DEFAULT_KB_ENTRIES)

    # Class method to create a MedicalKnowledgeBase from an official MeSH XML descriptor file
    @classmethod
    def from_mesh_descriptor_xml( cls, xml_path: str | Path, *, mesh_to_umls_map: str | Path | None = None, max_records: Optional[int] = None,
        include_terms: Sequence[str] | None = None,) -> "MedicalKnowledgeBase":
        # create KBEntries list from a mesh xml entry
        entries = load_mesh_entries( 
            xml_path, # path to the MeSH descriptor XML file
            mesh_to_umls_map=mesh_to_umls_map, # path to a JSON/TSV mapping of MeSH DescriptorUI -> UMLS CUI
            max_records=max_records, # maximum number of records to load and stop after (useful for prototyping)
            include_terms=include_terms, # list of terms to include only
        )

        # Return constructor function which return a MedicalKnowledgeBase object
        return cls.from_entries(entries)

    # Method to add an entry to the knowledge base (used in function from_entries).
    # Practically initialization function.
    def add_entry(self, entry: KBEntry) -> None:

        # add entry to entries dictionary
        self.entries[entry.concept_id] = entry

        # add alias to its MeSH corresponding alias_index dictionary
        for alias in entry.aliases():
            key = _normalize(alias) # apply private function normalize
            # if key is not empty assign it an alias_index which is the concept_id
            if key:
                self.alias_index[key] = entry.concept_id

    # Method to read KBEntry from its alias.
    def link(self, mention: str) -> Optional[KBEntry]:
        # get concept_id(MeSH ID) using its key(alias) from alias_index dictionary
        concept_id = self.alias_index.get(_normalize(mention))
        # Return KBEntry from its concept_id if it exists in entries otherwise return None
        return self.entries.get(concept_id) if concept_id else None

    def merge_entries(self, entries: Iterable[KBEntry | Dict[str, object]]) -> None:
        """Merge additional entries into the KB, overrides duplicates if they exist."""

        for entry in entries:
            if not isinstance(entry, KBEntry):
                entry = KBEntry(**entry)
            self.add_entry(entry)

# Create a class EntityExtraction from KBEntries
# @dataclass is a decorator that is used to create a class with automatic building of
# constructor functions and other methods.
@dataclass
class EntityExtraction:

    # Instantiate variables
    text: str
    label: str
    start_char: int
    end_char: int
    kb_entry: Optional[KBEntry] = None

    # Property to get the concept_id from the kb_entry
    @property
    def concept_id(self) -> Optional[str]:
        return self.kb_entry.concept_id if self.kb_entry else None

    # Property to get the canonical_name from the kb_entry
    @property
    def canonical_name(self) -> Optional[str]:
        return self.kb_entry.canonical_name if self.kb_entry else None

    # Property to get the entity_type from the kb_entry
    @property
    def entity_type(self) -> Optional[str]:
        return self.kb_entry.entity_type if self.kb_entry else None

    # Property to get the ontology from the kb_entry
    @property
    def ontology(self) -> Optional[str]:
        return self.kb_entry.ontology if self.kb_entry else None

    # Property to get the umls_cui from the kb_entry
    @property
    def umls_cui(self) -> Optional[str]:
        return self.kb_entry.cui if self.kb_entry else None


# Create a class MedspacyNERPipeline from KBEntries which combines the whole pipeline.

# @dataclass is a decorator that is used to create a class with automatic building of
# constructor functions and other methods.
@dataclass
class MedspacyNERPipeline:
    """Wrapper around medspaCy with optional rule bootstrapping and KB linking."""

    # instantiate variables
    knowledge_base: Optional[MedicalKnowledgeBase] = None
    add_default_rules: bool = True
    enable_linking: bool = True
    language: str = "en"
    nlp: Language = field(init=False)
    target_matcher: TargetMatcher = field(init=False)
    mesh_descriptor_path: Optional[str | Path] = None
    mesh_to_umls_map: Optional[str | Path] = None
    mesh_max_records: Optional[int] = None
    mesh_include_terms: Sequence[str] | None = None
    allowed_entity_types: Optional[Sequence[str]] = None
    _allowed_entity_types: Optional[Set[str]] = field(init=False, repr=False, default=None)

    # Initialize variables with values
    def __post_init__(self) -> None:
        self._ensure_knowledge_base()
        self._allowed_entity_types = (
            {etype.upper() for etype in self.allowed_entity_types}
            if self.allowed_entity_types
            else None
        )
        # Load medspacy pipeline without the following components
        self.nlp = medspacy.load(
            disable=["medspacy_context", "medspacy_sectionizer", "medspacy_postprocessor"]
        )
        # Get target matcher from nlp
        self.target_matcher = self.nlp.get_pipe("medspacy_target_matcher")
        # If add_default_rules is True, bootstrap rules from knowledge base
        if self.add_default_rules:
            self._bootstrap_rules_from_kb()

    # Private method to ensure knowledge base is initialized.
    def _ensure_knowledge_base(self) -> None:
        # If knowledge base and mesh_descriptor_path are not None load mesh entries into knowledge base
        if self.knowledge_base is not None:
            if self.mesh_descriptor_path:
                LOGGER.info("Merging MeSH descriptors into existing knowledge base")
                mesh_entries = load_mesh_entries(
                    self.mesh_descriptor_path,
                    mesh_to_umls_map=self.mesh_to_umls_map,
                    max_records=self.mesh_max_records,
                    include_terms=self.mesh_include_terms,
                )
                self.knowledge_base.merge_entries(mesh_entries)
            return
        # If knowledge base is None and mesh_descriptor_path is not None load mesh entries into a new knowledge base
        if self.mesh_descriptor_path:
            LOGGER.info("Initializing knowledge base from MeSH XML: %s", self.mesh_descriptor_path)
            self.knowledge_base = MedicalKnowledgeBase.from_mesh_descriptor_xml(
                self.mesh_descriptor_path,
                mesh_to_umls_map=self.mesh_to_umls_map,
                max_records=self.mesh_max_records,
                include_terms=self.mesh_include_terms,
            )
        # If knowledge base is None and mesh_descriptor_path is None load demo entries into new knowledge base    
        else:
            self.knowledge_base = MedicalKnowledgeBase.with_demo_entries() 

    # Private method to bootstrap rules from knowledge base.
    # bootstrapping is a technique of loading a program by means of a few initial instructions which enable 
    # the introduction of the rest of the program from an input device.
    # this is laying the foundation for the NER rules.
    def _bootstrap_rules_from_kb(self) -> None:
        # instantiate rules list
        rules: List[TargetRule] = [] # A TargetRule in medspaCy is a rule that says: “If you see this text pattern, treat it as an entity of type X.”

        # for loop to add all aliases and entry types to a rules list
        # + tokenize aliases at lowercase.
        for entry in self.knowledge_base.entries.values():
            for alias in entry.aliases():
                rules.append(
                    TargetRule(
                        literal=alias, # if the text matches the alias then it fits and so is the same type as the entity (e.g. "chest pain")
                        category=entry.entity_type, # the category is the entity type (e.g. "Disease")
                        pattern=[{"LOWER": token.lower()} for token in alias.split()], # the pattern matches the entity if it is not the exact literal text
                        # is the alias split into tokens (e.g. ["chest", "pain"])
                    )
                )

        # add rules to target matcher
        if rules:
            self.target_matcher.add(rules) # In medspaCy, the TargetMatcher is a pipeline component that:
            # Finds medical entities in text based on rules you define.

    # Public method to analyze a single document.
    def analyze(self, text: str) -> List[EntityExtraction]:
        doc = self.nlp(text)
        return self._entities_from_doc(doc) # Uses private method _entities_from_doc declared below to extract entity from document.

    # Public method to analyze multiple documents extract medical entities and optional KB-linked concepts via Medspacy pipeline..
    def analyze_documents(self, texts: Iterable[str]) -> List[List[EntityExtraction]]:
        return [self.analyze(text) for text in texts] # Uses anlyze method to extract entities from a group of documents.

    # Private method to extract entities from a document.
    def _entities_from_doc(self, doc) -> List[EntityExtraction]:
        entities: List[EntityExtraction] = []
        for ent in doc.ents:
            kb_entry = self.knowledge_base.link(ent.text) if self.enable_linking else None # Get KBEntry using entity text.
            if not self._is_allowed_entity(ent, kb_entry):
                continue
            # creating entity from knowledge base instance.
            entities.append(
                EntityExtraction(
                    text=ent.text,
                    label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    kb_entry=kb_entry,
                )
            )
        return entities # return list of entities

    # Public method to add a new rule to the target matcher.
    def add_rule(self, literal: str, label: str, **kwargs) -> None:
        """Append an ad-hoc target rule."""

        rule = TargetRule(literal=literal, category=label, **kwargs)
        self.target_matcher.add(rule)

    # Filter out unfiltered entities
    def _is_allowed_entity(self, ent, kb_entry: Optional[KBEntry]) -> bool:
        if not self._allowed_entity_types:
            return True
        if kb_entry and kb_entry.entity_type and kb_entry.entity_type.upper() in self._allowed_entity_types:
            return True
        label = (ent.label_ or "").upper()
        return label in self._allowed_entity_types


def load_mesh_entries(xml_path: str | Path, *, mesh_to_umls_map: str | Path | None = None, max_records: Optional[int] = None,
include_terms: Sequence[str] | None = None) -> List[KBEntry]:
    """Parse a MeSH Descriptor XML file into KBEntry objects.

    Parameters
    ----------
    xml_path:
        Path to `desc2024.xml` (or similar) downloaded from the MeSH FTP site.
    mesh_to_umls_map:
        Optional path to a JSON/TSV mapping of MeSH DescriptorUI -> UMLS CUI.
    max_records:
        Stop after loading this many descriptors (useful for prototyping).
    include_terms:
        If provided, only descriptors whose preferred name is in this collection are loaded.
    """

    # Ensure xml_path is a Path object and exists
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"MeSH descriptor XML not found: {xml_path}")

    # Load UMLS mapping if provided 
    umls_map = _load_mesh_umls_map(mesh_to_umls_map) if mesh_to_umls_map else {}
    include_filter = {term.lower() for term in include_terms} if include_terms else None # get lower case of include_terms

    # Parse XML file
    tree = ET.parse(xml_path) # resolves the sentence into its syntax parts
    root = tree.getroot() # returns the root element of the tree
    entries: List[KBEntry] = [] # list of entries

    # loop through all descriptor records in the XML file
    for record in root.findall("DescriptorRecord"):
        # get descriptor id of a record and its canonical name
        descriptor_id = (record.findtext("DescriptorUI") or "").strip()
        canonical = (record.findtext("DescriptorName/String") or "").strip()

        # if descriptor id or canonical name is empty, skip the record
        if not descriptor_id or not canonical:
            continue

        # if include_filter is not empty and canonical name is not in include_filter, skip the record
        if include_filter and canonical.lower() not in include_filter:
            continue

        #else extract synonyms, scope note, tree numbers, entity type, and cui
        synonyms = _extract_mesh_synonyms(record) # extract synonyms from record
        scope_note = (record.findtext("ScopeNote") or "").strip() or None # extract scope note from record
        tree_numbers = [tn.text for tn in record.findall("TreeNumberList/TreeNumber") if tn.text] # extract tree numbers from record
        entity_type = _guess_entity_type(tree_numbers) # extract entity type from tree numbers
        cui = umls_map.get(descriptor_id) # extract cui from umls_map using mesh_id as key

        # append KBEntry to entries list
        entries.append(
            KBEntry(
                concept_id=descriptor_id,
                canonical_name=canonical,
                entity_type=entity_type,
                synonyms=synonyms,
                description=scope_note,
                ontology="MeSH",
                cui=cui,
            )
        )

        # if max_records is not empty and number of entries is greater than or equal to max_records, break the loop
        if max_records and len(entries) >= max_records:
            break

    # log the number of entries loaded from an XML path
    LOGGER.info("Loaded %d MeSH descriptors from %s", len(entries), xml_path)
    return entries

# Function to load MeSH-UMLS mapping
def _load_mesh_umls_map(path: str | Path) -> Dict[str, str]:
    """Load MeSH-UMLS mapping from a JSON or TSV file."""

    # Convert JSON/TSV path to Path object
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MeSH-UMLS mapping file not found: {path}")

    # Read text from path
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}

    # Load JSON/TSV mapping
    if path.suffix.lower() == ".json": #if .suffix is .json then...
        data = json.loads(text) # load JSON data from text

        # if data is a dictionary then return the k and v of the data items as strings
        if isinstance(data, dict): 
            return {str(k): str(v) for k, v in data.items()}
        
        # else if data is a list then map as dictionary with string keys and values
        if isinstance(data, list):
            mapping: Dict[str, str] = {} # create empty dictionary where string is key

            # loop through data and extract mesh_id and cui
            for item in data:
                if isinstance(item, dict):
                    mesh_id = item.get("mesh") or item.get("mesh_id") or item.get("DescriptorUI")
                    cui = item.get("cui") or item.get("CUI")
                    if mesh_id and cui:
                        mapping[str(mesh_id)] = str(cui) # fill empty dictionary where mesh_id is the key and cui is the value
            return mapping
        raise ValueError("Unsupported JSON mapping format")


    # Now we go through the case where the file is a TSV file
    mapping = {} # create empty dictionary

    # loop through text and extract mesh_id and cui
    for line in text.splitlines():
        # skip empty lines and comments
        if not line.strip() or line.startswith("#"):
            continue
        
        # split line into mesh_id and cui
        parts = re.split(r"\s+", line.strip())
        
        # if parts is less than 2 then skip the line
        if len(parts) < 2:
            continue
        
        # extract mesh_id and cui
        mesh_id, cui = parts[0], parts[1]
        
        # fill empty dictionary where mesh_id is the key and cui is the value
        mapping[mesh_id] = cui
    return mapping


# Function to extract MeSH synonyms from a record
def _extract_mesh_synonyms(record: ET.Element) -> List[str]:
    terms = []
    
    # loop through record and extract MeSH synonyms
    for term in record.findall("ConceptList/Concept/TermList/Term/String"):
        # if term.text is not empty then append it to terms list
        if term.text:
            value = term.text.strip()
            if value:
                terms.append(value)
    # unduplicate while preserving order
    seen = set()
    unique_terms = []
    for term in terms:
        key = term.lower()
        
        # if key is not in seen then add it to seen and append it to unique_terms list
        if key not in seen:
            seen.add(key)
            unique_terms.append(term)
    return unique_terms

# Function to guess entity type from tree numbers
def _guess_entity_type(tree_numbers: List[str]) -> str:
    """Map MeSH tree numbers to coarse entity types (disease, symptom, etc.)."""

    if not tree_numbers:
        return "ENTITY"

    # store first letter of each element in tree_numbers
    prefixes = {tn[0] for tn in tree_numbers if tn}

    # if prefixes is a subset of {"C", "F"} then return "DISEASE"
    if prefixes & {"C", "F"}:
        return "DISEASE"
    
    # if prefixes is a subset of {"A"} then return "ANATOMY"
    if prefixes & {"A"}:
        return "ANATOMY"
    
    # if prefixes is a subset of {"D"} then return "CHEMICAL"
    if prefixes & {"D"}:
        return "CHEMICAL"
    
    # if prefixes is a subset of {"E"} then return "PROCEDURE"
    if prefixes & {"E"}:
        return "PROCEDURE"
    
    # else return "ENTITY"
    return "ENTITY"

# private list of default dictionary knowledge base entries
_DEFAULT_KB_ENTRIES = [
    {
        "concept_id": "MESH:D010146",
        "canonical_name": "Pneumonia",
        "entity_type": "DISEASE",
        "synonyms": ["lung infection", "pulmonary infection"],
        "description": "Inflammation of the lungs primarily due to infection.",
    },
    {
        "concept_id": "MESH:D012140",
        "canonical_name": "Pain",
        "entity_type": "SYMPTOM",
        "synonyms": ["ache", "painful sensation", "chest pain"],
        "description": "An unpleasant sensory and emotional experience.",
    },
    {
        "concept_id": "MESH:D020820",
        "canonical_name": "Nausea",
        "entity_type": "SYMPTOM",
        "synonyms": ["queasiness", "stomach sickness"],
        "description": "A feeling of sickness with an inclination to vomit.",
    },
    {
        "concept_id": "MESH:D007052",
        "canonical_name": "Ibuprofen",
        "entity_type": "MEDICATION",
        "synonyms": ["advil", "motrin"],
        "description": "A nonsteroidal anti-inflammatory drug.",
    },
    {
        "concept_id": "MESH:D003920",
        "canonical_name": "Diabetes Mellitus",
        "entity_type": "DISEASE",
        "synonyms": ["diabetes", "dm"],
        "description": "Metabolic disorders characterized by hyperglycemia.",
    },
]


__all__ = [
    "KBEntry",
    "MedicalKnowledgeBase",
    "EntityExtraction",
    "MedspacyNERPipeline",
]
