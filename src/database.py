import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import logging

from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import settings

logger = logging.getLogger(__name__)

# Initialize components
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index = pc.Index(settings.PINECONE_INDEX_NAME)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=1536,
    google_api_key=settings.GEMINI_API_KEY
)

def query_vector_db(query: str, top_k: int = 3):
    """Queries Pinecone and returns matched documents alongside their similarity scores."""
    query_vector = embeddings.embed_query(query)
    
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    logger.info("=" * 40)
    logger.info(f"🔍 RAW PINECONE MATCHES FOR QUERY: '{query}'")
    if response.matches:
        for i, match in enumerate(response.matches):
            logger.info(f"  Match [{i+1}]:")
            logger.info(f"    ↳ ID: {match.id}")
            logger.info(f"    ↳ Similarity Score: {match.score:.4f}")

            text_preview = match.metadata.get('text', '')[:150].replace('\n', ' ')
            logger.info(f"    ↳ Metadata Text Preview: {text_preview}...")
    else:
        logger.info("  ❌ No matches found in Pinecone.")
    logger.info("=" * 40)
    
    docs = []
    max_score = 0.0
    
    if response.matches:
        max_score = response.matches[0].score
        for match in response.matches:
            docs.append(match.metadata.get("text", ""))
            
    return docs, max_score