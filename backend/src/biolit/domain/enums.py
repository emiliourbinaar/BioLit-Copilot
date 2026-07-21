from enum import StrEnum


class Source(StrEnum):
    pubmed = "pubmed"
    biorxiv = "biorxiv"
    medrxiv = "medrxiv"


class TextType(StrEnum):
    full_text_available = "full_text_available"
    full_text_unverified = "full_text_unverified"
    abstract_only = "abstract_only"


class LicenseTier(StrEnum):
    open = "open"
    non_commercial = "non_commercial"
    restricted = "restricted"
    unknown = "unknown"
