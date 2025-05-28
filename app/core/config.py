from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    OPENROUTER_API_KEY: str
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    LLM_MODEL_NAME: str = "mistralai/mistral-7b-instruct-v0.2" # Comece com um modelo confiável
    FAISS_INDEX_PATH: str = "faiss_index_multi_author"

    # Para carregar do arquivo .env
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8', extra='ignore')

settings = Settings()