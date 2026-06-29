import argparse
import logging
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from graph import clean_graph
from pipeline.build_graph import run
from graph.schema_profiles import ExtractionMode, SchemaLevel, parse_property_spec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Neo4j knowledge graph from ChromaDB chunks."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete all existing graph data before running.",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="ChromaDB collection name (overrides .env CHROMA_COLLECTION).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of chunks per LLM extraction batch (overrides .env BATCH_SIZE).",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="Optional Chroma namespace filter (overrides .env CHROMA_NAMESPACE).",
    )
    parser.add_argument(
        "--pages",
        type=parse_pages,
        default=None,
        help="Optional comma-separated page_no filter, such as 5,17,21.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["groq", "local"],
        default=None,
        help="Choose groq or local OpenAI-compatible LLM provider.",
    )
    parser.add_argument(
        "--schema-level",
        choices=[level.value for level in SchemaLevel],
        default=None,
        help=(
            "Extraction schema preset. "
            "Choose unconstrained, constrained, or strict."
        ),
    )
    parser.add_argument(
        "--schema-profile",
        default=None,
        help="Path to a YAML schema profile for constrained or strict extraction.",
    )
    parser.add_argument(
        "--extraction-mode",
        choices=[mode.value for mode in ExtractionMode],
        default=None,
        help="Choose tool for structured output or prompt for fallback extraction.",
    )
    parser.add_argument(
        "--node-properties",
        default=None,
        help=(
            "Node property mode: off, any, or a comma-separated list "
            "such as birth_date,death_date."
        ),
    )
    parser.add_argument(
        "--relationship-properties",
        default=None,
        help=(
            "Relationship property mode: off, any, or a comma-separated list "
            "such as start_date."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.clean:
        print("Cleaning graph …")
        clean_graph()
        print("Graph cleared.")

    print("Starting pipeline …")
    stats = run(
        collection_name=args.collection,
        batch_size=args.batch_size,
        namespace=args.namespace,
        pages=args.pages,
        llm_provider=args.llm_provider,
        schema_level=args.schema_level,
        schema_profile_path=args.schema_profile,
        extraction_mode=args.extraction_mode,
        node_properties=(
            parse_property_spec(args.node_properties)
            if args.node_properties is not None
            else None
        ),
        relationship_properties=(
            parse_property_spec(args.relationship_properties)
            if args.relationship_properties is not None
            else None
        ),
    )

    print(
        f"\nDone — {stats.chunks_processed} chunks → "
        f"{stats.graph_docs_created} graph docs in {stats.elapsed_seconds:.1f}s"
    )
    if stats.errors:
        print(f"{len(stats.errors)} batch error(s) — check logs above.")
        sys.exit(1)

def parse_pages(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    pages = []
    for part in value.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        try:
            page = int(stripped)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid page number '{stripped}'"
            ) from exc
        if page <= 0:
            raise argparse.ArgumentTypeError("Page numbers must be positive integers")
        pages.append(page)
    return tuple(sorted(set(pages))) or None


if __name__ == "__main__":
    main()
