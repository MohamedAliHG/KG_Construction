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
    local_llm_api_key: str | None = "local-key"
    local_llm_model: str = "local-model"

    # ChromaDB
    chroma_path: str = "./chroma_db_demo"
    chroma_collection: str = "collection_demo"

    # Pipeline
    batch_size: int = 10
    extraction_mode: str = "tool"
    schema_level: str = "unconstrained"
    node_properties: str = "off"
    relationship_properties: str = "off"



settings = Settings()
