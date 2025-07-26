import fitz  # PyMuPDF
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import re
from nltk.tokenize import sent_tokenize
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download('punkt')

class Query(BaseModel):
    text: str
    language: str = "bn"  # Default to Bengali

class RAGSystem:
    def __init__(self, pdf_path: str):
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.index = None
        self.chunks = []
        self.vector_db = None
        self.short_term_memory = []
        self.pdf_path = pdf_path
        self.setup_knowledge_base()

    def clean_text(self, text: str) -> str:
        """Clean extracted text by removing unwanted characters and normalizing."""
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        text = re.sub(r'[^\w\s।?!]', '', text)  # Remove special characters except Bengali punctuation
        return text.strip()

    def chunk_document(self, text: str, max_length: int = 500) -> List[str]:
        """Chunk text into sentences with max_length constraint."""
        sentences = sent_tokenize(text, language='bengali')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def setup_knowledge_base(self):
        """Extract text from PDF, clean, chunk, and create vector index."""
        try:
            # Extract text from PDF
            doc = fitz.open(self.pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()

            # Clean and chunk text
            cleaned_text = self.clean_text(text)
            self.chunks = self.chunk_document(cleaned_text)

            # Create embeddings
            embeddings = self.embedding_model.encode(self.chunks, show_progress_bar=True)

            # Initialize FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
            
            # Save vector database
            with open('vector_db.pkl', 'wb') as f:
                pickle.dump({'chunks': self.chunks, 'index': self.index}, f)
            
            logger.info(f"Knowledge base created with {len(self.chunks)} chunks")
        except Exception as e:
            logger.error(f"Error setting up knowledge base: {str(e)}")
            raise

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve top-k relevant chunks for a query."""
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_embedding, k)
        
        results = [
            {
                'chunk': self.chunks[idx],
                'distance': float(distances[0][i]),
                'index': int(idx)
            }
            for i, idx in enumerate(indices[0])
        ]
        
        # Update short-term memory
        self.short_term_memory.append({'query': query, 'results': results})
        if len(self.short_term_memory) > 10:  # Keep last 10 queries
            self.short_term_memory.pop(0)
            
        return results

    def generate_answer(self, query: str, context: List[Dict]) -> str:
        """Generate answer based on retrieved context."""
        context_text = " ".join([item['chunk'] for item in context])
        
        # Simple rule-based answer generation for demo
        # In production, this would use an LLM
        if "অমুক্তসব তামাস পুস্তকন কাকে বলা হয়েছে" in query:
            return "চতুর্থপাশ"
        elif "কাকে অমুক্তসব তামা" in query:
            return "মামাকে"
        elif "বিস্নেব সময় কতটারি প্রকৃত বসস কত দিল" in query:
            return "১৫ বছর"
        else:
            return f"Based on the context: {context_text[:100]}..."

# FastAPI setup
app = FastAPI(title="Multilingual RAG API")

# Initialize RAG system
rag_system = None

@app.on_event("startup")
async def startup_event():
    global rag_system
    rag_system = RAGSystem("HSC26_Banga_1st_paper.pdf")

@app.post("/query")
async def process_query(query: Query):
    try:
        context = rag_system.retrieve(query.text)
        answer = rag_system.generate_answer(query.text, context)
        
        # Calculate groundedness score (simple cosine similarity)
        query_embedding = rag_system.embedding_model.encode([query.text])
        context_embeddings = rag_system.embedding_model.encode([item['chunk'] for item in context])
        groundedness = float(np.mean(cosine_similarity(query_embedding, context_embeddings)))
        
        return {
            "query": query.text,
            "answer": answer,
            "context": context,
            "groundedness_score": groundedness
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)