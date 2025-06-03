# IA Multi Author Backend

This repository contains a small FastAPI server that exposes a RAG (Retrieval Augmented Generation) endpoint using a FAISS vector index. A preprocessing script is provided to create the index from PDFs.

## Required dependencies

Python 3.11 is recommended. Install the Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Some packages (such as `faiss-cpu` and `PyMuPDF`) may require system build tools or additional libraries on your platform.

## Environment variables

The application reads its settings from environment variables (or a `.env` file):

- `OPENROUTER_API_KEY` – API key used for the language model.
- `EMBEDDING_MODEL_NAME` – name of the sentence transformer embedding model.
- `LLM_MODEL_NAME` – model identifier used by the chat completion endpoint.
- `FAISS_INDEX_PATH` – directory containing the FAISS index (defaults to `faiss_index_multi_author`).

## Running the preprocessing script

Place your source PDFs inside the `pdf_sources/` directory and provide metadata in `pdf_metadata.json`. Then run:

```bash
python preprocess_and_create_index.py
```

The script splits the PDFs, generates embeddings and stores the FAISS index under `FAISS_INDEX_PATH`.

## Starting the API

### Using Docker

A `dockerfile` is included. Build and run the container with:

```bash
docker build -t ia-multi-author .
docker run -p 8000:8000 ia-multi-author
```

The FAISS index directory will be copied into the image if it exists when building.

### Using Uvicorn locally

Activate your virtual environment and run:

```bash
uvicorn app.main:app --reload
```

By default the server listens on port `8000`.

## Configuring CORS

The allowed origins are defined in `app/main.py` in the `origins` list. Adjust it to include the URLs of your front‑end applications before deploying.

## Supplying an external FAISS index

If the index folder is not bundled with the source code, set `FAISS_INDEX_PATH` to the directory where the `index.faiss` and `index.pkl` files can be found (for example by mounting a volume or copying the files at startup).

