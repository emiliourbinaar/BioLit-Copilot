"""Build the MeSH descriptor tree-number artifact from the NLM descriptor dump.

Mirrors `build_mesh`: read a raw dump already on disk, write a compact gzipped artifact that
the runtime loads. No network. The source `data/mesh/desc2026.gz` was downloaded for the
ADR-0011 work and is gitignored, as is the output.

`main()` gets no direct unit test per project convention; `build_tree_table` and `MeshTree`
are tested in their own module.
"""

DEFAULT_SOURCE = "data/mesh/desc2026.gz"


def main(argv: list[str] | None = None) -> None:
    import argparse
    import gzip
    from pathlib import Path

    from biolit.canon.mesh_tree import MeshTree, build_tree_table
    from biolit.config import get_settings

    parser = argparse.ArgumentParser(description="Build the MeSH tree-number artifact.")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    out = args.out or get_settings().mesh_tree_artifact_path
    with gzip.open(args.source, "rt", encoding="utf-8", errors="replace") as fh:
        table = build_tree_table(fh.read())
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    MeshTree(table).save_artifact(out)
    placements = sum(len(v) for v in table.values())
    print(f"{len(table)} descriptors, {placements} tree placements -> {out}")


if __name__ == "__main__":
    main()
