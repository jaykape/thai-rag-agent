import httpx
import respx
from fastapi.testclient import TestClient

from app.main import app, OLLAMA_URL, API_KEY

client = TestClient(app)
AUTH = {"Authorization": f"Bearer {API_KEY}"}
BODY = {"model": "qwen3.5:9b", "messages": [{"role": "user", "content": "hi"}]}


def test_health_needs_no_key():
    assert client.get("/health").json() == {"ok": True}


def test_missing_key_is_401():
    assert client.post("/v1/chat/completions", json=BODY).status_code == 401


def test_wrong_key_is_401():
    r = client.post("/v1/chat/completions", json=BODY,
                    headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


@respx.mock
def test_chat_proxies_to_ollama():
    respx.post(f"{OLLAMA_URL}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": "สวัสดี"}}]})
    )
    r = client.post("/v1/chat/completions", json=BODY, headers=AUTH)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "สวัสดี"
