import hmac
import os

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
API_KEY = os.getenv("API_KEY", "dev-key")

app = FastAPI(title="Thai RAG Agent")


def require_key(request: Request) -> None:
    supplied = request.headers.get("authorization", "")
    if not hmac.compare_digest(supplied, f"Bearer {API_KEY}"):
        raise HTTPException(status_code=401, detail="invalid key")


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/v1/models", dependencies=[Depends(require_key)])
async def models():
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{OLLAMA_URL}/v1/models")
    return JSONResponse(r.json(), status_code=r.status_code)


@app.post("/v1/chat/completions", dependencies=[Depends(require_key)])
async def chat(request: Request):
    body = await request.json()
    url = f"{OLLAMA_URL}/v1/chat/completions"

    if body.get("stream"):
        async def stream():
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", url, json=body) as r:
                    async for chunk in r.aiter_raw():
                        yield chunk

        return StreamingResponse(stream(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(url, json=body)
    return JSONResponse(r.json(), status_code=r.status_code)
