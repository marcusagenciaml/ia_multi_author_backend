from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any  # Any é usado para permitir diferentes tipos na página

class ChatRequest(BaseModel):
    query: str
    # Opcional: Adicionar um campo para o usuário especificar um autor, se quisermos essa funcionalidade no futuro
    # author_filter: Optional[str] = None 

class SourceDocument(BaseModel):
    content: str
    page: Optional[Any] = None # Mantido como Any
    author: Optional[str] = None
    book_title: Optional[str] = None

class ChatResponse(BaseModel):
    answer: Optional[str] = None
    source_documents: List[SourceDocument] = Field(default_factory=list)
    error: Optional[str] = None
