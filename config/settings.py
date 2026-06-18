from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Neo4j (local instance)
    neo4j_url: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str

    # Groq
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"

    # Local OpenAI-compatible LLM server
    llm_provider: str = "groq"
    local_llm_base_url: str = "http://localhost:8080/v1"
    local_llm_api_key: str | None = None
    local_llm_model: str = "local-model"

    # ChromaDB
    chroma_path: str = "./chroma_db"
    chroma_collection: str = "collection_demo"
    chroma_namespace: str | None = "default"

    # KG pipeline
    batch_size: int = 10
    extraction_mode: str = "tool"
    schema_level: str = "unconstrained"
    node_properties: str = "off"
    relationship_properties: str = "off"

    # Document indexing pipeline
    document_input_path: str = "data/raw"
    document_chunk_strategy: str = "hybrid"
    document_image_resolution_scale: float = 2.0
    document_enable_picture_description: bool = True
    document_vlm_url: str = "http://localhost:8080"
    document_vlm_model_name: str = "qwen"
    document_vlm_timeout: int = 60
    document_vlm_prompt: str = "Describe this image in sentences in a single paragraph."
    document_fixed_character_chunk_size: int = 2000
    document_fixed_character_chunk_overlap: int = 200
    document_hierarchical_merge_list_items: bool = True
    document_hybrid_tokenizer: str = "BAAI/bge-small-en-v1.5"
    document_hybrid_max_tokens: int = 450
    document_hybrid_merge_peers: bool = True
    document_hybrid_image_mode: str = "placeholder"
    document_hybrid_image_placeholder: str = ""
    document_hybrid_mark_annotations: bool = True
    document_hybrid_include_annotations: bool = True
    document_embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    document_embedding_device: str = "cpu"
    document_embedding_normalize_embeddings: bool = True
    document_chroma_reset_collection: bool = False



settings = Settings()
