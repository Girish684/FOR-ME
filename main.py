from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import uuid
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load your Groq key
load_dotenv()

app = FastAPI(title="My AI Agent - Final Corrected Version")

# === STRONG CORS FIX ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq Client
client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Chat memory
conversations = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str = None
    model: str = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = "You are a helpful and friendly AI assistant."

@app.get("/")
async def root():
    print("✅ Someone visited the root page")
    return {"status": "✅ Backend is LIVE and READY! 🚀", "message": "Go to index.html and test"}

@app.post("/api/chat")
async def chat_stream(request: ChatRequest):
    print(f"📥 Received request: {request.message}")

    if not request.session_id:
        request.session_id = str(uuid.uuid4())

    if request.session_id not in conversations:
        conversations[request.session_id] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    # Add user message
    conversations[request.session_id].append({"role": "user", "content": request.message})
    print(f"📝 Session {request.session_id} - User message added")

    async def generate():
        full_response = ""
        try:
            # Clean messages (this fixes the timestamp error)
            clean_messages = [
                {"role": m["role"], "content": m["content"]}
                for m in conversations[request.session_id]
            ]

            print("🚀 Sending request to Groq...")
            stream = await client.chat.completions.create(
                model=request.model,
                messages=clean_messages,
                temperature=0.7,
                max_tokens=2048,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield f"data: {json.dumps({'type': 'chunk', 'content': content})}\n\n"

            # Save reply
            conversations[request.session_id].append({"role": "assistant", "content": full_response})
            print("✅ Groq replied successfully")
            yield f"data: {json.dumps({'type': 'end', 'content': full_response})}\n\n"

        except Exception as e:
            error_text = str(e)
            print(f"❌ ERROR: {error_text}")
            yield f"data: {json.dumps({'type': 'error', 'message': error_text})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

# Run the server
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI Agent Backend on http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)