# KG Construction

A modular knowledge graph construction pipeline that:

- reads text chunks from a local ChromaDB collection
- extracts entities and relationships with an LLM
- writes the graph into a local Neo4j database

The project supports:

- three schema levels: `unconstrained`, `constrained`, `strict`
- two extraction modes: `tool` and `prompt`
- node and relationship property extraction in tool mode
- Groq or a local OpenAI-compatible LLM endpoint

## Project Layout

- `config/` - runtime settings
- `ingestion/` - ChromaDB loading helpers
- `graph/` - transformer, schema presets, and Neo4j persistence
- `pipeline/` - end-to-end orchestration
- `scripts/` - command-line entry points
- `tests/` - unit tests

## Requirements

- Python 3.11+
- Neo4j running locally
- ChromaDB data available locally
- A Groq API key or a local OpenAI-compatible LLM server

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root.

### Neo4j

```env
neo4j_url=bolt://localhost:7687
neo4j_username=neo4j
neo4j_password=your_password
```

### LLM Provider

#### Groq

```env
llm_provider=groq
groq_api_key=your_groq_key
groq_model=llama-3.3-70b-versatile
```

#### Local OpenAI-Compatible Server

```env
llm_provider=local
local_llm_base_url=http://localhost:8080/v1
local_llm_model=your-local-model-name
local_llm_api_key=local-key
```

### ChromaDB

```env
chroma_path=./chroma_db
chroma_collection=collection_demo
```

### Pipeline Defaults

```env
batch_size=10
schema_level=unconstrained
extraction_mode=tool
node_properties=off
relationship_properties=off
```

## CLI Usage

Run the full pipeline:

```bash
python scripts/run_pipeline.py
```

Common options:

- `--clean` - delete all existing Neo4j graph data before running
- `--collection NAME` - choose a ChromaDB collection
- `--batch-size N` - number of chunks per batch
- `--llm-provider groq|local` - choose the model backend
- `--schema-level unconstrained|constrained|strict` - choose schema strictness
- `--extraction-mode tool|prompt` - choose structured or prompt-based extraction
- `--node-properties off|any|prop1,prop2,...` - configure node property extraction
- `--relationship-properties off|any|prop1,prop2,...` - configure relationship property extraction

Examples:

```bash
python scripts/run_pipeline.py --clean
python scripts/run_pipeline.py --collection collection_demo --batch-size 5
python scripts/run_pipeline.py --llm-provider local --schema-level strict
python scripts/run_pipeline.py --schema-level constrained --extraction-mode prompt
python scripts/run_pipeline.py --schema-level strict --node-properties any
python scripts/run_pipeline.py --schema-level strict --node-properties birth_date,death_date --relationship-properties start_date
```

## Extraction Modes

### Schema Levels

- `unconstrained` - no schema constraints
- `constrained` - allowed nodes plus flat allowed relationships
- `strict` - allowed nodes plus source/type/target relationship tuples

### LLM Modes

- `tool` - uses structured extraction when the model supports tools
- `prompt` - fallback prompt-based extraction

### Property Extraction

Property extraction is only supported in `tool` mode.

- `off` - do not extract properties
- `any` - let the model choose useful properties
- `prop1,prop2,...` - extract only the listed properties

## Data Flow

1. `ingestion.load_chunks()` reads chunks from ChromaDB
2. `graph.transformer.extract_graph_documents()` converts chunks to `GraphDocument` objects
3. `graph.neo4j_store.add_graph_documents()` writes the graph into Neo4j

The pipeline processes batches sequentially to keep LLM usage predictable.


## Testing

Run the test suite with:

```bash
.venv/bin/pytest -q
```

The current test suite covers:

- ChromaDB loading
- schema profile presets
- tool vs prompt transformer wiring
- property extraction settings
- pipeline argument threading

## Troubleshooting

### Missing Groq Key

If `llm_provider=groq`, you must set `groq_api_key`.

### Local Model Not Available

If `llm_provider=local`, make sure the local server is reachable at the configured base URL and that it exposes an OpenAI-compatible API.

### Neo4j Connection Errors

Confirm the Neo4j instance is running and the password in `.env` is correct.

### Empty Graph Output

Check that:

- the Chroma collection exists
- the collection contains chunks
- the LLM endpoint is reachable
- the selected schema level is not too restrictive

## Suggested First Run

```bash
python scripts/run_pipeline.py --llm-provider groq --schema-level unconstrained
```

If the first run succeeds, try narrowing the schema and enabling properties:

```bash
python scripts/run_pipeline.py --llm-provider groq --schema-level strict --node-properties any
```
