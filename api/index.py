from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage

from src.models import ChatRequest, ChatResponse
from src.workflow import app_graph

app = FastAPI(title="Blue Cross RAG Backend API", version="1.0")

# Enable CORS for React frontend hosting
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    try:
        config = {"configurable": {"thread_id": payload.thread_id}}
        
        # Execute state machine via checkpointer session tracking
        output = app_graph.invoke(
            {"messages": [HumanMessage(content=payload.message)]}, 
            config=config
        )
        
        return ChatResponse(response=output["final_response"])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}