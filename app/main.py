# app/main.py

from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.v1.endpoints.router import api_router
from app.services.rag_service import initialize_rag_components # Apenas importe a função
# NÃO importe qa_chain_global aqui: from app.services.rag_service import qa_chain_global
from app.core.config import settings
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Aplicação iniciando...")
    logger.info(f"Carregando com OPENROUTER_API_KEY: {'*' * (len(settings.OPENROUTER_API_KEY) - 4) + settings.OPENROUTER_API_KEY[-4:] if settings.OPENROUTER_API_KEY else 'NÃO DEFINIDA'}")
    logger.info(f"Modelo de Embedding: {settings.EMBEDDING_MODEL_NAME}")
    logger.info(f"Modelo LLM: {settings.LLM_MODEL_NAME}")
    logger.info(f"Caminho do Índice FAISS: {settings.FAISS_INDEX_PATH}")
    
    try:
        initialize_rag_components() # Chamar a inicialização
        # A função initialize_rag_components já loga sucesso ou falha.
        # Se houver uma falha crítica lá, ela pode até levantar uma exceção
        # para impedir o startup, se desejado.
        logger.info("Inicialização dos componentes RAG solicitada. Verifique os logs do rag_service para o status detalhado.")
    except Exception as e:
        logger.critical(f"Falha catastrófica durante a inicialização dos componentes RAG: {e}", exc_info=True)
        # Aqui você poderia decidir parar a aplicação se a inicialização falhar completamente
        # raise RuntimeError("Falha crítica na inicialização do RAG.") from e
    
    yield
    logger.info("Aplicação encerrando...")

app = FastAPI(
    title="IA Multi-Autor RAG API",
    description="Uma API para interagir com uma IA baseada no conhecimento de múltiplos autores, usando RAG.",
    version="0.1.0",
    lifespan=lifespan
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/", tags=["Root"], summary="Endpoint raiz da API")
async def root():
    # Para verificar o status real, o ideal seria que o rag_service expusesse uma função de status
    # Por agora, esta é uma mensagem genérica.
    # Se você testar o endpoint /ask e ele funcionar, a chain está ok.
    return {
        "message": "Bem-vindo à API da IA Multi-Autor!",
        "rag_status": "Inicialização tentada. Verifique os logs do rag_service ou teste o endpoint /ask.",
        "docs_url": "/docs"
    }