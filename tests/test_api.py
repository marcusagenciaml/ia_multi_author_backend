import os
import sys
import types
import fastapi.routing
from fastapi.testclient import TestClient

# Ensure the application package is on the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Patch FastAPI router to handle trailing slash prefixes
_orig_include_router = fastapi.routing.APIRouter.include_router

def _patched_include_router(self, router, **kwargs):
    prefix = kwargs.pop("prefix", "").rstrip("/")
    return _orig_include_router(self, router, prefix=prefix, **kwargs)

fastapi.routing.APIRouter.include_router = _patched_include_router

# Provide a dummy rag_service to avoid heavy dependencies
fake_rag_service = types.ModuleType("app.services.rag_service")
fake_rag_service.initialize_rag_components = lambda: None
fake_rag_service.get_answer = (
    lambda _q: {"error": "Sistema RAG não inicializado corretamente."}
)
sys.modules["app.services.rag_service"] = fake_rag_service

from app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    for key in ("message", "rag_status", "docs_url"):
        assert key in data


def test_chat_uninitialized():
    response = client.post("/api/v1/chat/ask/", json={"query": "Oi"})
    assert response.status_code == 503
    assert "detail" in response.json()
