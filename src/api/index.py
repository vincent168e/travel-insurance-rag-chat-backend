import sys
import logging
from pathlib import Path
from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
import cloudinary
import cloudinary.uploader

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.models import ChatRequest, ChatResponse
from src.workflow import app_graph
from src.config import settings

logging.basicConfig(level=logging.INFO)

cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET,
    secure=True
)

app = FastAPI(title="Cobalt Cross Travel Insurance RAG Chat Backend API", version="1.0")

# Enable CORS for React frontend hosting
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
            settings.LOCAL_FRONTEND_CLIENT_URL,
            settings.EXTERNAL_FRONTEND_CLIENT_URL
        ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    try:
        config = {"configurable": {"thread_id": payload.thread_id}}
        
        # Explicitly reset checkpoint state for inquiry switches
        if payload.service_category == "inquiry" and payload.claim_stage is None:
            try:
                # Force updates the checkpointer state to clear stale claim stages
                app_graph.update_state(config, {"claim_stage": None})
            except Exception:
                # Catch cases where the thread/checkpoint doesn't exist yet
                pass
        
        # Execute state machine via checkpointer session tracking
        output = app_graph.invoke(
            {
                "messages": [HumanMessage(content=payload.message)],
                "user_input": payload.message,
                "service_category": payload.service_category,
                "claim_category": payload.claim_category,
                "claim_description": payload.claim_description,
                "claim_stage": payload.claim_stage,
                "image_urls": payload.image_urls,
            },
            config=config
        )
        
        return ChatResponse(
            thread_id=payload.thread_id,
            response=output.get("final_response", ""),
            service_category=output.get("service_category"),
            claim_category=output.get("claim_category"),
            claim_description=output.get("claim_description"),
            session_closed=bool(output.get("session_closed", False)),
            claim_stage=output.get("claim_stage"),
            audit_report=output.get("audit_report"),
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_images_endpoint(files: List[UploadFile] = File(...)):
    """
    Accepts multiple raw file binaries, uploads each to Cloudinary in a loop,
    and returns a list of public secure URL strings.
    """
    uploaded_urls = []
    
    for file in files:
        try:
            # Upload individual file stream to Cloudinary
            upload_result = cloudinary.uploader.upload(
                file.file,
                folder="travel-insurance-rag-chat/attachments",
            )
            
            secure_url = upload_result.get("secure_url")
            if secure_url:
                uploaded_urls.append(secure_url)
                
        except Exception as e:
            # If one fails, you can choose to skip it or raise an exception
            logging.error(f"Failed uploading file '{file.filename}': {str(e)}")

            raise HTTPException(
                status_code=500, 
                detail=f"Failed uploading file '{file.filename}': {str(e)}"
            )
            
    return {"image_urls": uploaded_urls}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}