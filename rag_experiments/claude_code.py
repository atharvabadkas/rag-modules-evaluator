import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
import warnings
import numpy as np

warnings.filterwarnings("ignore")

class EnhancedRAGApp:
    def __init__(self, csv_path: str):
        """Initialize enhanced RAG application with improved data processing and chunking"""
        load_dotenv()
        self.api_key = os.getenv('OPENAI_API_KEY')
        
        # Initialize text splitter for better chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        print("Loading dataset...")
        self.load_and_process_data(csv_path)
        self.setup_components()
        self.create_vector_store()

    def clean_and_format_value(self, val: Any) -> str:
        """Clean and format values for better text representation"""
        if pd.isna(val):
            return ""
        if isinstance(val, (int, float)):
            if float(val).is_integer():
                return f"{int(val):,}"
            return f"{val:,.2f}"
        return str(val).strip()

    def create_structured_content(self, row: pd.Series, idx: int) -> str:
        """Create more structured and readable content for each record"""
        sections = []
        
        # Basic information
        basic_info = [f"Record ID: {idx + 1}"]
        
        # Group related fields
        field_groups = {}
        for col, val in row.items():
            cleaned_val = self.clean_and_format_value(val)
            if cleaned_val:
                # You can customize these groupings based on your data
                if 'date' in col.lower():
                    field_groups.setdefault('Temporal Information', []).append(f"{col}: {cleaned_val}")
                elif any(term in col.lower() for term in ['price', 'cost', 'revenue', 'amount']):
                    field_groups.setdefault('Financial Information', []).append(f"{col}: {cleaned_val}")
                else:
                    field_groups.setdefault('General Information', []).append(f"{col}: {cleaned_val}")
        
        # Add grouped information
        for group_name, fields in field_groups.items():
            sections.append(f"{group_name}:\n" + "\n".join(fields))
        
        return "\n\n".join(sections)

    def load_and_process_data(self, csv_path: str):
        """Load and process the dataset with improved data handling"""
        # Load data with optimized settings
        self.df = pd.read_csv(csv_path, low_memory=False)
        
        # Clean column names
        self.df.columns = self.df.columns.str.lower().str.strip()
        
        # Improve data type conversion
        for col in self.df.columns:
            # Try numeric conversion
            if self.df[col].dtype == 'object':
                try:
                    self.df[col] = pd.to_numeric(self.df[col], errors='ignore')
                except:
                    pass
                    
            # Clean string columns
            if self.df[col].dtype == 'object':
                self.df[col] = self.df[col].str.strip()

        # Create enhanced documents
        print("Processing documents...")
        self.documents = []
        
        for idx, row in self.df.iterrows():
            content = self.create_structured_content(row, idx)
            
            # Create document with enhanced metadata
            doc = Document(
                page_content=content,
                metadata={
                    'row_id': idx,
                    'record_number': idx + 1,
                    **{k: v for k, v in row.items() if pd.notna(v)}
                }
            )
            self.documents.append(doc)
        
        # Split documents into smaller chunks for better retrieval
        self.documents = self.text_splitter.split_documents(self.documents)

    def setup_components(self):
        """Setup LangChain components with improved prompting"""
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.api_key,
            model="text-embedding-3-large"  # Using the latest embedding model
        )
        
        self.llm = ChatOpenAI(
            temperature=0,
            model_name='gpt-4-turbo-preview',  # Using GPT-4 for better reasoning
            openai_api_key=self.api_key
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a precise data analyst assistant specialized in retrieving and analyzing information from structured datasets. Your responses should be accurate, well-organized, and based solely on the provided context.

            Critical Guidelines:
            1. Always analyze ALL provided context thoroughly before responding
            2. For numerical questions, perform exact calculations and show your work
            3. When asked to list or rank items, examine ALL records in the context
            4. Include specific record numbers and exact values in your responses
            5. If information is missing or unclear, explicitly state this
            6. Format numerical values with appropriate commas and decimal places
            7. Organize your response in a clear, structured manner
            8. If asked about trends or patterns, provide specific examples
            
            The data is organized in records with the following structure:
            - Record ID: Unique identifier for each entry
            - Temporal Information: Date-related fields
            - Financial Information: Price, cost, and monetary fields
            - General Information: Other relevant fields
            
            Context: {context}"""),
            ("human", "{question}"),
            ("system", "Provide a comprehensive answer using ALL relevant information from the context. Include specific record numbers and calculations where appropriate.")
        ])

    def create_vector_store(self):
        """Create vector store with improved settings"""
        print("Creating vector store...")
        self.vector_store = FAISS.from_documents(
            documents=self.documents,
            embedding=self.embeddings
        )
        print("Vector store created successfully!")

    def ask(self, question: str, k: int = 15) -> str:
        """Process a question with improved retrieval and answer generation"""
        try:
            # Get more relevant documents with MMR retrieval
            docs = self.vector_store.max_marginal_relevance_search(
                question,
                k=k,
                fetch_k=k*2  # Fetch more documents initially for better filtering
            )
            
            # Combine context with clear separation
            context = "\n\n=== Next Record ===\n\n".join(doc.page_content for doc in docs)
            
            # Get response
            chain = self.prompt | self.llm
            response = chain.invoke({
                "context": context,
                "question": question
            })
            
            return response.content.strip()
            
        except Exception as e:
            return f"Error processing question: {str(e)}\nPlease try rephrasing your question or contact support."

    def get_column_stats(self, column: str) -> dict:
        """Get comprehensive statistics about a specific column"""
        if column not in self.df.columns:
            return {"error": f"Column '{column}' not found in dataset"}
            
        col_data = self.df[column]
        stats = {
            "column_name": column,
            "data_type": str(col_data.dtype),
            "total_rows": len(col_data),
            "null_count": col_data.isna().sum(),
            "unique_values_count": len(col_data.unique())
        }
        
        if pd.api.types.is_numeric_dtype(col_data):
            stats.update({
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "mean": float(col_data.mean()),
                "median": float(col_data.median()),
                "std": float(col_data.std()),
                "quartiles": {
                    "25%": float(col_data.quantile(0.25)),
                    "50%": float(col_data.quantile(0.50)),
                    "75%": float(col_data.quantile(0.75))
                }
            })
        else:
            value_counts = col_data.value_counts()
            stats.update({
                "top_10_values": value_counts.head(10).to_dict(),
                "sample_values": col_data.dropna().sample(min(5, len(col_data))).tolist()
            })
            
        return stats

# Example usage
if __name__ == "__main__":
    rag = EnhancedRAGApp('atharva-prep-dataset - master-data.csv')
    
    # Example questions
    questions = [
        "What are the top 10 SKUs by prep contribution?"
    ]
    
    print("\n=== Processing Questions ===\n")
    for question in questions:
        print(f"Question: {question}")
        print("\nAnalyzing...\n")
        answer = rag.ask(question)
        print(f"Answer: {answer}")
        print("\n" + "="*50 + "\n")