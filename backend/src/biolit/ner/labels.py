CHEMICAL = "CHEMICAL"
DISEASE = "DISEASE"

_CANONICAL: dict[str, str] = {"chemical": CHEMICAL, "disease": DISEASE}


def canonical_label(raw: str) -> str | None:
    """Map a model tag (Chemical, disease, B-Chemical, I-Disease, ...) to a canonical label.

    Strips any BIO prefix and matches case-insensitively; unknown tags return None.
    """
    if not raw:
        return None
    key = raw.split("-")[-1].strip().lower()
    return _CANONICAL.get(key)
