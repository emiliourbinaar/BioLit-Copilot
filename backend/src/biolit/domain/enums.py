from enum import Enum


class Source(str, Enum):
    pubmed = "pubmed"
    biorxiv = "biorxiv"
    medrxiv = "medrxiv"


class TextType(str, Enum):
    full_text_available = "full_text_available"
    full_text_unverified = "full_text_unverified"
    abstract_only = "abstract_only"


class LicenseTier(str, Enum):
    open = "open"
    non_commercial = "non_commercial"
    restricted = "restricted"
    unknown = "unknown"
