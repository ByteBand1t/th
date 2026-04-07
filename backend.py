class ChatRequest(BaseModel):
    ip: str
    port: int
    model: str
    message: str
    history: list = []

@app.post("/api/chat")
def chat_with_model(req: ChatRequest):
    """Send a single message to an Ollama model and return the response."""
    import time
    base_url = f"http://{req.ip}:{req.port}"
    messages = req.history + [{"role": "user", "content": req.message}]
    payload  = {"model": req.model, "messages": messages, "stream": False}
    t0 = time.perf_counter()
    try:
        r = httpx.post(f"{base_url}/api/chat", json=payload, timeout=60.0)
        r.raise_for_status()
        elapsed  = time.perf_counter() - t0
        response = r.json().get("message", {}).get("content", "")
        return {"response": response, "elapsed": round(elapsed, 2)}
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Model timed out")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.post("/api/add-anyway")
def add_anyway(req: AddRequest):
    """Force-add a failed model to LiteLLM regardless of benchmark results."""
    req_dict = req.dict()
    return add_manual(AddRequest(**req_dict))
