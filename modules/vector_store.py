from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import torch
from typing import List, Dict
import logging
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import gc

# Configure simple logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.setup_device()
        self.setup_embeddings()

    def setup_device(self):
        """Setup the appropriate device for M1 Mac"""
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        logger.info(f"Using device: {self.device}")

    def setup_embeddings(self):
        """Initialize embeddings with optimizations"""
        try:
            # Basic model configuration
            model_kwargs = {
                'device': self.device
            }

            # Separate encoding configuration
            encode_kwargs = {
                'batch_size': 32,
                'normalize_embeddings': True
            }

            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            logger.info("Embeddings model initialized successfully")
        except Exception as e:
            logger.error(f"Error setting up embeddings: {e}")
            raise

    def initialize(self, df: pd.DataFrame):
        """Initialize or load vector store"""
        try:
            if self._load_from_cache():
                return

            logger.info("Creating new vector store...")
            documents = self._create_documents(df)
            
            # Clear memory before creating FAISS index
            gc.collect()
            if self.device == 'mps':
                torch.mps.empty_cache()
            
            self.store = FAISS.from_documents(
                documents,
                self.embeddings
            )
            
            self._save_to_cache()
            logger.info("Vector store created and saved successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise

    def _create_documents(self, df: pd.DataFrame) -> List[Document]:
        """Create documents with optimized batch processing"""
        documents = []
        batch_size = 50
        
        try:
            total_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)
            
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i + batch_size]
                batch_docs = []
                
                for _, row in batch.iterrows():
                    # Clean and format row data
                    row_dict = {
                        k: str(v).strip() if pd.notna(v) else "N/A" 
                        for k, v in row.items()
                    }
                    
                    # Create structured content
                    content_parts = []
                    for k, v in row_dict.items():
                        if v != "N/A":
                            content_parts.append(f"{k}: {v}")
                    
                    content = " | ".join(content_parts)
                    doc = Document(page_content=content, metadata=row_dict)
                    batch_docs.append(doc)
                
                documents.extend(batch_docs)
                
                # Memory management
                if i % (batch_size * 10) == 0:
                    gc.collect()
                    if self.device == 'mps':
                        torch.mps.empty_cache()
            
            return documents
            
        except Exception as e:
            logger.error(f"Error creating documents: {e}")
            raise

    def _load_from_cache(self) -> bool:
        """Load vector store from cache"""
        cache_path = self.cache_dir / "vector_store.faiss"
        try:
            if cache_path.exists():
                logger.info("Loading vector store from cache...")
                self.store = FAISS.load_local(str(self.cache_dir), self.embeddings)
                logger.info("Vector store loaded successfully")
                return True
        except Exception as e:
            logger.warning(f"Failed to load vector store from cache: {e}")
            if cache_path.exists():
                logger.info("Removing corrupted cache...")
                cache_path.unlink()
        return False

    def _save_to_cache(self):
        """Save vector store to cache"""
        try:
            self.store.save_local(str(self.cache_dir))
            logger.info("Vector store cached successfully")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")

    def search(self, query: str, k: int = 3) -> List[Document]:
        """Optimized search"""
        try:
            # Clear memory before search
            if self.device == 'mps':
                torch.mps.empty_cache()
            
            results = self.store.similarity_search(
                query,
                k=k
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return [] 