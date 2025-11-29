"""Utilities for downloading the dataset and preprocessing medical NER text."""

import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional
import kagglehub

try:
    import spacy
    from spacy.language import Language
except ImportError:  # pragma: no cover - guards optional dependency at runtime
    spacy = None
    Language = None  # type: ignore[assignment]

# Prepared operations and variables
DATASET_ID = "tsunmm/maccrobat2018"
DATA_DIR = "C:/Users/laith/Medical_NER/data"
_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "kagglehub", "datasets", DATASET_ID.replace("/", os.sep)
) # Path to the cached dataset
_WHITESPACE_RE = re.compile(r"\s+") # Regular expression to match whitespace

# Function to download the dataset if needed
def ensure_dataset(data_dir: str = DATA_DIR) -> str:
    """Download the Kaggle dataset if needed and return the local data directory."""

    if not os.path.exists(_CACHE_DIR):
        path = kagglehub.dataset_download(DATASET_ID) #download dataset
        os.makedirs(data_dir, exist_ok=True) # Create data directory
        shutil.copytree(path, data_dir, dirs_exist_ok=True) # Copy dataset to data directory
        print("Path to dataset files:", path)
    else:
        print("Path to dataset files:", data_dir)
    return data_dir

# Create a class for PreprocessingPipeline
@dataclass
class PreprocessingPipeline:
    """Simple text preprocessing pipeline for medical NER datasets.

    The pipeline performs:
      1. Text cleaning (case handling, removal of irrelevant characters, whitespace fix)
      2. Tokenization via spaCy (default: lightweight blank English tokenizer)

    Example
    -------
    >>> pipeline = PreprocessingPipeline()
    >>> pipeline.preprocess("Patient reports Chest Pain & nausea!!!")
    {'cleaned_text': 'patient reports chest pain nausea',
     'tokens': ['patient', 'reports', 'chest', 'pain', 'nausea']}
    """

    # variables for preprocessing
    lowercase: bool = True # a variable lowercase of type bool which is set to True
    keep_chars: str = ".,;:/-()%" # a variable keep_chars of type str which is set to ".,;:/-()%"
    tokenizer: Optional["Language"] = None # a variable tokenizer of type Optional["Language"] 
    # this means it can either be of type Language as imported from spacy library or of type None.
    # Optional is a type hint that indicates that the variable can be None.
    _clean_pattern: re.Pattern = field(init=False, repr=False) # a variable _clean_pattern of type re.Pattern.
    # init=False means that this variable is not initialized in the constructor.  This means:
    #Python will not accept _clean_pattern as a parameter in the dataclass constructor.

    # The variable must be created inside __post_init__, not passed in by the user.
    # repr=False means that this variable is not included in the string representation of the object.

    # The __post_init__ method is called after the constructor has been called.
    def __post_init__(self) -> None:
        allowed = re.escape(self.keep_chars) # a variable "allowed" of type str which is set to ".,;:/-()%"
        self._clean_pattern = re.compile(rf"[^0-9A-Za-z\s{allowed}]") # fill variable "_clean_pattern" with a regular expression pattern. 
        # which matches any character that is not a digit, a letter, a whitespace, or a character in "allowed".
        if self.tokenizer is None:
            # If spacy library not imported properly raise error
            if spacy is None:
                raise ImportError(
                    "spaCy is required for tokenization. Please install the dependencies "
                    "listed in requirements.txt."
                )
            self.tokenizer = spacy.blank("en") # otherwise fill variable "tokenizer" with a blank English tokenizer (empty pipeline — no tagger, no parser, no NER.)

    # The clean_text method is called to clean the text.
    def clean_text(self, text: str) -> str:
        """Normalize casing, drop noisy characters, and collapse extra whitespace."""

        # Strip leading and trailing whitespace
        cleaned = text.strip()

        # Normalize all characters to lowercase
        if self.lowercase:
            cleaned = cleaned.lower()
        
        # Replace any character that is not a digit, a letter, a whitespace, or a character in "allowed" with a space
        cleaned = self._clean_pattern.sub(" ", cleaned)

        # Replace any whitespace with a space
        cleaned = _WHITESPACE_RE.sub(" ", cleaned)
        return cleaned.strip()

    # The tokenize method is called to tokenize the text.
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using the configured spaCy tokenizer."""

        # If tokenizer is not established
        if not self.tokenizer:
            raise RuntimeError("Tokenizer is not initialized.")

        # else tokenize the text
        doc = self.tokenizer(text)
        return [token.text for token in doc if token.text.strip()] # return token.text for index token which goes through doc if token.text.strip() is not just an 
        # empty string.

    # The preprocess method is called to clean and tokenize a single string, returning both views.
    def preprocess(self, text: str) -> Dict[str, List[str] | str]:
        """Clean and tokenize a single string, returning both views."""

        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        return {"cleaned_text": cleaned, "tokens": tokens}

    # The preprocess_corpus method is called to clean and tokenize multiple strings, returning both views.
    def preprocess_corpus(self, texts: Iterable[str]) -> List[Dict[str, List[str] | str]]:
        """Apply the pipeline to multiple documents."""

        return [self.preprocess(text) for text in texts]


if __name__ == "__main__":
    # Eagerly ensure the dataset is ready when the module is imported in scripts.
    ensure_dataset()


