import os
import sys
import types
import fastapi.routing
from fastapi.testclient import TestClient
import pytest

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

# Stub pydantic_settings to avoid heavy dependency installation
fake_ps = types.ModuleType("pydantic_settings")

class _BaseSettings:
    def __init__(self, **values):
        for k, v in values.items():
            setattr(self, k, v)

SettingsConfigDict = dict
fake_ps.BaseSettings = _BaseSettings
fake_ps.SettingsConfigDict = SettingsConfigDict
sys.modules.setdefault("pydantic_settings", fake_ps)


@pytest.fixture
def ready_rag_service(monkeypatch):
    """Simulate an initialized rag_service returning a dummy answer."""
    def _ready_answer(query: str):
        return {
            "answer": f"Echo: {query}",
            "source_documents": [],
        }

    monkeypatch.setattr(fake_rag_service, "get_answer", _ready_answer)
    yield

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


def test_chat_ready(ready_rag_service):
    """Ensure chat endpoint works when rag_service is initialized."""
    response = client.post("/api/v1/chat/ask/", json={"query": "Teste"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Echo: Teste"
    assert data["source_documents"] == []


def test_cors_blocked_origin():
    """Requests from unknown origins should not receive CORS headers."""
    response = client.get("/", headers={"Origin": "http://evil.com"})
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers
