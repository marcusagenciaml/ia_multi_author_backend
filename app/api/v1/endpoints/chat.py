from fastapi import APIRouter, HTTPException, Body
from app.services import rag_service
from app.models_pydantic.chat import ChatRequest, ChatResponse
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/ask/", response_model=ChatResponse, summary="Faz uma pergunta à IA")
async def ask_question(request_body: ChatRequest = Body(..., example={"query": "Qual o segredo para uma vida plena?"})):
    """
    Recebe uma pergunta do usuário e retorna a resposta da IA baseada no RAG,
    juntamente com os documentos fonte utilizados.
    """
    if not request_body.query or not request_body.query.strip():
        raise HTTPException(status_code=400, detail="A pergunta (query) não pode ser vazia.")
    
    logger.info(f"Endpoint /ask chamado com query: {request_body.query[:100]}")
    response_data = rag_service.get_answer(request_body.query)
    
    if response_data.get("error"):
        logger.error(f"Erro retornado pelo rag_service: {response_data['error']}")
        # Em vez de 500 genérico, podemos ser mais específicos se o erro indicar
        # Por exemplo, se o erro for "Sistema RAG não inicializado", poderia ser um 503 Service Unavailable
        if "não inicializado" in response_data["error"]:
             raise HTTPException(status_code=503, detail=response_data["error"])
        raise HTTPException(status_code=500, detail=response_data["error"])
            
    return ChatResponse(**response_data)