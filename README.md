# ia_multi_author_backend

This project exposes a FastAPI application for a Retrieval-Augmented Generation (RAG) service.

## Configuration

Settings are loaded from environment variables using `pydantic-settings`.  The most relevant variables are:

- `OPENROUTER_API_KEY` – API key used by the RAG service.
- `EMBEDDING_MODEL_NAME` – name of the embedding model.
- `LLM_MODEL_NAME` – name of the language model.
- `FAISS_INDEX_PATH` – directory containing the FAISS index.
- `ALLOWED_ORIGINS` – comma-separated list of origins allowed by CORS (e.g. `"http://localhost:3000,https://example.com"`).

Create a `.env` file or export the variables before starting the application.
