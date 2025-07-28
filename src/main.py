from fastapi import FastAPI, HTTPException, Depends, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import uvicorn
from typing import Optional
import json
import structlog

from db.database import get_db
from agents.agent_graph import build_agent_graph
from privacy.masking import DatabaseMasker, DocumentMasker
from db.vector_store import FAISSVectorStore

logger = structlog.get_logger()

app = FastAPI(title="Secure Agentic RAG System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
db_masker = DatabaseMasker()
doc_masker = DocumentMasker()
vector_store = FAISSVectorStore()
agent_graph = build_agent_graph(db_masker, doc_masker, vector_store)

@app.post("/query")
async def process_query(
    query: str,
    session_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Process natural language queries with privacy protection"""
    try:
        state = {
            "query": query,
            "session_id": session_id,
            "db_session": db
        }
        
        result = agent_graph.run(state)
        return {
            "answer": result["final_answer"],
            "session_id": result.get("session_id")
        }
    except Exception as e:
        logger.error("Query processing error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload and process documents for vector storage"""
    try:
        # Read and process document
        content = await file.read()
        
        # Mask sensitive information
        masked_content, mapping = doc_masker.mask_document(content)
        
        # Store in vector database
        doc_id = vector_store.add_document(masked_content)
        
        # Store mapping for later retrieval
        db_masker.store_mapping(doc_id, mapping, db)
        
        return {"status": "success", "document_id": doc_id}
    except Exception as e:
        logger.error("Document upload error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
