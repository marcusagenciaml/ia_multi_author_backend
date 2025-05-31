# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # <<---- 1. IMPORTE O MIDDLEWARE
from contextlib import asynccontextmanager
from app.api.v1.endpoints.router import api_router
from app.services.rag_service import initialize_rag_components
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
        initialize_rag_components()
        logger.info("Inicialização dos componentes RAG solicitada. Verifique os logs do rag_service para o status detalhado.")
    except Exception as e:
        logger.critical(f"Falha catastrófica durante a inicialização dos componentes RAG: {e}", exc_info=True)
    
    yield
    logger.info("Aplicação encerrando...")

app = FastAPI(
    title="IA Multi-Autor RAG API",
    description="Uma API para interagir com uma IA baseada no conhecimento de múltiplos autores, usando RAG.",
    version="0.1.0",
    lifespan=lifespan
)

# --- CONFIGURAÇÃO DO CORS ---
# <<---- 2. DEFINA AS ORIGENS PERMITIDAS ---->>
origins = [
    "http://localhost:3000",  # Seu frontend Next.js rodando localmente para desenvolvimento
    # Adicione aqui a URL do seu frontend quando fizer deploy na Vercel
    # Ex: "https://seu-projeto-frontend.vercel.app"
    # Adicione também "https://ai.agenciaml.com" se você planeja ter um frontend servido no mesmo domínio
    # ou se outras aplicações nesse domínio precisarem acessar a API.
    # Por segurança, seja o mais específico possível com as origens.
]

# <<---- 3. ADICIONE O MIDDLEWARE CORS À APLICAÇÃO ---->>
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Lista de origens que têm permissão para fazer requisições
    allow_credentials=True, # Permite cookies em requisições cross-origin (se você usar)
    allow_methods=["*"],    # Permite todos os métodos (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],    # Permite todos os cabeçalhos
)
# --- FIM DA CONFIGURAÇÃO DO CORS ---

app.include_router(api_router, prefix="/api/v1/")

@app.get("/", tags=["Root"], summary="Endpoint raiz da API")
async def root():
    return {
        "message": "Bem-vindo à API da IA Multi-Autor!",
        "rag_status": "Inicialização tentada. Verifique os logs do rag_service ou teste o endpoint /api/v1/chat/ask.", # Ajuste o endpoint de teste
        "docs_url": "/docs"
    }