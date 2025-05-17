from .data_processor import DataProcessor
from .vector_store import VectorStore
from .llm import LLM
import logging
from typing import Optional
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_time=False)]
)
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, csv_path: str, api_key: str, cache_dir: str = "cache"):
        """Initialize the RAG system"""
        # Initialize components
        self.data_processor = DataProcessor(csv_path, cache_dir)
        self.vector_store = VectorStore(cache_dir)
        self.llm = LLM(api_key)
        
        # Initialize vector store with processed data
        self.vector_store.initialize(self.data_processor.df)
        
        # Setup LLM chain with vector store retriever
        self.llm.setup_chain(self.vector_store.store.as_retriever(
            search_kwargs={"k": 5}
        ))
        
        logger.info("RAG system initialized successfully")

    def ask(self, question: str) -> str:
        """Process a question and return an answer"""
        try:
            # Get statistics if needed
            stats = self.data_processor.statistics if self._needs_stats(question) else {}
            
            # Get response from LLM
            response = self.llm.get_response(
                question=question,
                context=[],  # Context will be handled by the chain
                stats=stats,
                columns=self.data_processor.statistics['columns']
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return f"Error: {str(e)}"

    def _needs_stats(self, question: str) -> bool:
        """Check if question likely needs statistical information"""
        stat_keywords = ['average', 'mean', 'maximum', 'minimum', 'top', 'highest', 'lowest', 'median', 'total']
        return any(keyword in question.lower() for keyword in stat_keywords) 