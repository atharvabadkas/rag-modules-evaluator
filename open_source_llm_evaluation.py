from typing import Dict, List
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import time

# Set up the embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

class LLMEvaluator:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.vector_index = None
        self.df = None
        
    def add_model(self, name: str, model_name: str):
        """Add a local Ollama model to the evaluation suite"""
        llm = OllamaLLM(
            model=model_name,
            timeout=300  # 5 minutes timeout
        )
        self.models[name] = llm
        
    def setup_models(self):
        """Initialize different local LLM models"""
        # You can add any model you've pulled with Ollama
        self.add_model("llama2", "llama2")
        self.add_model("mistral", "mistral")
        self.add_model("phi", "phi")
        
    def load_csv(self, csv_path: str):
        """Load data from CSV file"""
        self.df = pd.read_csv(csv_path)
        print(f"Loaded CSV with {len(self.df)} rows and columns: {', '.join(self.df.columns)}")
        
        # Convert DataFrame to documents
        documents = []
        
        # Create a document for each row
        for idx, row in self.df.iterrows():
            # Convert row to string representation
            row_text = "Row " + str(idx + 1) + ":\n"
            for col in self.df.columns:
                row_text += f"{col}: {row[col]}\n"
            documents.append(Document(text=row_text))
            
        # Create a schema document
        schema_text = "Dataset Schema:\n"
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            schema_text += f"- {col} ({dtype})\n"
        documents.append(Document(text=schema_text))
        
        # Create summary document
        summary_text = "Dataset Summary:\n"
        summary_text += f"- Total rows: {len(self.df)}\n"
        summary_text += f"- Columns: {', '.join(self.df.columns)}\n"
        
        # Add basic statistics for numeric columns
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            summary_text += "\nNumeric Column Statistics:\n"
            for col in numeric_cols:
                summary_text += f"{col}:\n"
                summary_text += f"  - Mean: {self.df[col].mean():.2f}\n"
                summary_text += f"  - Min: {self.df[col].min():.2f}\n"
                summary_text += f"  - Max: {self.df[col].max():.2f}\n"
        
        documents.append(Document(text=summary_text))
        
        return documents
        
    def create_vector_index(self, csv_path: str = None, documents: List[str] = None, model_name: str = "llama2"):
        """Create a vector index from CSV file or documents"""
        if csv_path:
            docs = self.load_csv(csv_path)
        elif documents:
            docs = [Document(text=doc) for doc in documents]
        else:
            raise ValueError("Either csv_path or documents must be provided")
            
        # Set up LLM in global settings with timeout
        Settings.llm = Ollama(model=model_name, timeout=120)  # Increase timeout to 120 seconds
        
        # Create vector index
        self.vector_index = VectorStoreIndex.from_documents(docs)
        
    def query_index(self, model_name: str, query: str) -> str:
        """Query the vector index using specified model"""
        if not self.vector_index:
            raise ValueError("Vector index not created. Call create_vector_index first.")
            
        # Update LLM in settings for current query
        Settings.llm = Ollama(model=model_name)
            
        query_engine = self.vector_index.as_query_engine()
        response = query_engine.query(query)
        return str(response)
        
    def evaluate_model(self, model_name: str, questions: List[str], use_index: bool = False, prompt_template: str = None):
        """Evaluate a single model on a set of questions"""
        if prompt_template is None:
            prompt_template = """
            You are an AI assistant analyzing a dataset. Answer the following question based on the data provided:
            Question: {question}
            Answer: """
            
        prompt = PromptTemplate(
            input_variables=["question"],
            template=prompt_template
        )
        
        llm = self.models[model_name]
        chain = prompt | llm | StrOutputParser()
        
        results = []
        
        for question in questions:
            start_time = time.time()
            try:
                if use_index:
                    response = self.query_index(model_name, question)
                else:
                    response = chain.invoke({"question": question})
                elapsed = time.time() - start_time
                results.append({
                    "question": question,
                    "response": response,
                    "time": elapsed,
                    "mode": "vector_index" if use_index else "direct",
                    "error": None
                })
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                print(f"Error with {model_name} on question '{question}': {error_type} - {error_msg}")
                results.append({
                    "question": question,
                    "response": "Failed to generate response",
                    "time": time.time() - start_time,
                    "mode": "vector_index" if use_index else "direct",
                    "error": f"{error_type}: {error_msg}"
                })
                
        self.results[model_name] = results
        return results
    
    def evaluate_all_models(self, questions: List[str], use_index: bool = False, prompt_template: str = None):
        """Evaluate all models on the dataset"""
        for model_name in self.models:
            print(f"Evaluating {model_name}...")
            self.evaluate_model(model_name, questions, use_index, prompt_template)
            
    def display_results(self):
        """Display results in a readable format in the terminal"""
        for model_name, results in self.results.items():
            print(f"\n{'='*80}")
            print(f"Results for {model_name.upper()}:")
            print(f"{'='*80}")
            
            for result in results:
                print(f"\nQuestion: {result['question']}")
                print(f"Mode: {result['mode']}")
                print(f"Response time: {result['time']:.2f}s")
                if result.get('error'):
                    print(f"Error: {result['error']}")
                print(f"Response:")
                print(f"{result['response']}")
                print(f"{'-'*80}")

# Example usage
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = LLMEvaluator()
    evaluator.setup_models()
    
    # Load and index your CSV file
    print("Creating vector index from CSV...")
    evaluator.create_vector_index(csv_path="atharva-prep-dataset - master-data.csv")
    
    # Specific data lookup question
    questions = [
        "What is the maximum total weight recorded for the vessel sandwich counter?",
    ]
    
    # Custom prompt template optimized for data lookup
    custom_prompt = """
    You are an AI assistant analyzing a manufacturing dataset. The dataset contains:
    - Timestamps for each entry
    - Item weights
    - Ingredient names (like shira)
    - Vessel information
    
    Please look through the data and find the exact item weight for the following query:
    Question: {What was the item weight of sheera on 5th January 2025?}
    
    If you find the entry, provide:
    1. The exact item weight
    2. The vessel used (if available)
    3. The exact timestamp
    
    If you cannot find an exact match, please say so explicitly.
    
    Answer: """
    
    # Skip direct questioning as it won't have access to the actual data
    print("\nEvaluating models with vector index...")
    evaluator.evaluate_all_models(questions, use_index=True, prompt_template=custom_prompt)
    
    # Display results in terminal
    evaluator.display_results()