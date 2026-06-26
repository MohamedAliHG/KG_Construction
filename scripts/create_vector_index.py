"""Create the Neo4j vector index for Document embeddings.

Verification query:

```cypher
CALL db.index.vector.queryNodes(
  'document_embedding_index',
  5,
  $query_embedding
)
YIELD node, score
MATCH (node)-[:MENTIONS]->(e:Entity)
RETURN node.id, node.text, score, collect(e.name) AS entities
LIMIT 5;
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph import create_document_vector_index

DEFAULT_DIMENSIONS = 384


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Neo4j vector index for :Document embeddings."
    )
    parser.add_argument(
        "--index-name",
        default="document_embedding_index",
        help="Neo4j vector index name.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=DEFAULT_DIMENSIONS,
        help="Embedding dimension (default: 384).",
    )
    parser.add_argument(
        "--similarity-function",
        default="cosine",
        choices=["cosine", "euclidean", "dot_product"],
        help="Vector similarity function.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_document_vector_index(
        index_name=args.index_name,
        dimensions=args.dimensions,
        similarity_function=args.similarity_function,
    )
    print(
        "Created Neo4j vector index "
        f"'{args.index_name}' (dimensions={args.dimensions}, similarity={args.similarity_function})"
    )


if __name__ == "__main__":
    main()
