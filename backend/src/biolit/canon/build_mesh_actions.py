"""Build the MeSH pharmacological-action artifact from the NLM descriptor dump.

Mirrors `build_mesh_tree`: read a raw dump already on disk, write a compact gzipped artifact
that the runtime loads. No network. The source `data/mesh/desc2026.gz` was downloaded for the
ADR-0011 work and is gitignored, as is the output -- so this is the same dump `build_mesh_tree`
reads, a second relation extracted from it rather than a second download.

`main()` gets no direct unit test per project convention; `build_action_table` and
`PharmacologicalActions` are tested in their own module.
"""

DEFAULT_SOURCE = "data/mesh/desc2026.gz"


def main(argv: list[str] | None = None) -> None:
    import argparse
    import gzip
    from pathlib import Path

    from biolit.canon.mesh_actions import PharmacologicalActions, build_action_table
    from biolit.config import get_settings

    parser = argparse.ArgumentParser(description="Build the MeSH pharmacological-action artifact.")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    out = args.out or get_settings().mesh_actions_artifact_path
    with gzip.open(args.source, "rt", encoding="utf-8", errors="replace") as fh:
        table = build_action_table(fh.read())
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    PharmacologicalActions(table).save_artifact(out)
    memberships = sum(len(v) for v in table.values())
    print(f"{len(table)} descriptors, {memberships} class memberships -> {out}")


if __name__ == "__main__":
    main()
