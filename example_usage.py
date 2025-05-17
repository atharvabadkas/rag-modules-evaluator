from rag.rag_system import RAGSystem
import time
import logging
import os
from dotenv import load_dotenv

# Configure simple logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    try:
        print("\n=== RAG System Initialization ===\n")
        
        # Print basic system info
        print(f"Python Version: {os.sys.version.split()[0]}")
        print(f"Cache Directory: ./rag_cache")
        print()
        
        # Load environment variables
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY', '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')
        
        # Initialize RAG system
        print("Initializing RAG system...")
        start_time = time.time()
        
        rag = RAGSystem(
            csv_path='atharva-prep-dataset - master-data.csv',
            api_key=api_key,
            cache_dir="./rag_cache"
        )
        
        print(f"Initialization complete in {time.time() - start_time:.2f} seconds\n")
        
        # Process questions
        questions = [
            "list top 10 skus with the highest weights",
        ]

        print("=== Processing Questions ===\n")
        
        for question in questions:
            try:
                print(f"Question: {question}")
                start_time = time.time()
                response = rag.ask(question)
                end_time = time.time()
                
                print("\nAnswer:")
                print(response)
                print(f"\nResponse time: {end_time - start_time:.2f} seconds")
                print("\n" + "="*50 + "\n")
                
            except Exception as e:
                print(f"Error processing question: {str(e)}\n")

    except Exception as e:
        print(f"Error during initialization: {str(e)}")

if __name__ == "__main__":
    main() 