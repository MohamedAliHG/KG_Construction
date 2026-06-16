import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from config import settings

TRUNCATE_AT = 120


def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=settings.chroma_path)


def cmd_list_collections(client: chromadb.PersistentClient) -> None:
    collections = client.list_collections()
    if not collections:
        print("No collections found.")
        return
    print(f"Found {len(collections)} collection(s) in '{settings.chroma_path}':\n")
    for col in collections:
        count = client.get_collection(col.name).count()
        print(f"  • {col.name}  ({count} chunks)")


def cmd_show_chunks(
    client: chromadb.PersistentClient,
    collection_name: str,
    limit: int | None,
    full: bool,
    search: str | None,
) -> None:
    try:
        col = client.get_collection(collection_name)
    except Exception:
        print(f"Collection '{collection_name}' not found.")
        print("Run with --collections to see available collections.")
        sys.exit(1)

    total = col.count()
    fetch_limit = limit or total

    result = col.get(
        limit=fetch_limit,
        include=["documents", "metadatas", "embeddings"],
    )

    ids = result["ids"]
    docs = result["documents"]
    metas = result["metadatas"]
    has_embeddings = result.get("embeddings") is not None

    # Filter by keyword if requested
    if search:
        filtered = [
            (i, d, m)
            for i, d, m in zip(ids, docs, metas)
            if search.lower() in d.lower()
        ]
        print(f"Collection : {collection_name}")
        print(f"Total      : {total} chunks")
        print(f"Search     : '{search}' → {len(filtered)} match(es)\n")
        ids, docs, metas = zip(*filtered) if filtered else ([], [], [])
    else:
        shown = len(ids)
        print(f"Collection : {collection_name}")
        print(f"Total      : {total} chunks")
        print(f"Showing    : {shown}{'' if not limit else f' of {total}'}")
        print(f"Embeddings : {'yes' if has_embeddings else 'no'}\n")

    sep = "─" * 72

    for idx, (chunk_id, text, meta) in enumerate(zip(ids, docs, metas), 1):
        print(sep)
        print(f"[{idx}] ID: {chunk_id}")
        if meta:
            for k, v in meta.items():
                print(f"     {k}: {v}")
        if full:
            print(f"\n{text}\n")
        else:
            preview = text.replace("\n", " ")
            if len(preview) > TRUNCATE_AT:
                preview = preview[:TRUNCATE_AT] + " …"
            print(f"\n{preview}\n")

    print(sep)
    print(f"\n{len(ids)} chunk(s) displayed.  Use --full to see complete text.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect chunks stored in a ChromaDB collection."
    )
    parser.add_argument(
        "--collections",
        action="store_true",
        help="List all available collections and their chunk counts.",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help=f"Collection name to inspect (default: {settings.chroma_collection!r}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of chunks to display.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print the full chunk text instead of a truncated preview.",
    )
    parser.add_argument(
        "--search",
        default=None,
        help="Only show chunks containing this keyword (case-insensitive).",
    )
    parser.add_argument(
        "--path",
        default=None,
        help=f"Override ChromaDB path (default: {settings.chroma_path!r}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.path:
        settings.chroma_path = args.path

    client = get_client()

    if args.collections:
        cmd_list_collections(client)
        return

    collection_name = args.collection or settings.chroma_collection
    cmd_show_chunks(
        client=client,
        collection_name=collection_name,
        limit=args.limit,
        full=args.full,
        search=args.search,
    )


if __name__ == "__main__":
    main()