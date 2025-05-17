from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import pandas as pd
from typing import List
import os

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

class RAGSystem:
    def __init__(self, filepath: str):
        self.df = pd.read_csv(filepath)
        # Convert date columns to datetime
        for col in self.df.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                except:
                    pass
        self.chain = self._create_chain()
    
    def _prepare_documents(self) -> List[Document]:
        documents = []
        
        # Add schema information
        schema = "Dataset Schema:\n"
        for col in self.df.columns:
            schema += f"{col} ({self.df[col].dtype})\n"
        documents.append(Document(page_content=schema, metadata={"type": "schema"}))
        
        # Process rows in chunks
        chunk_size = 100
        for i in range(0, len(self.df), chunk_size):
            chunk = self.df.iloc[i:i + chunk_size]
            
            # Group related rows together
            for _, row in chunk.iterrows():
                # Format datetime values
                formatted_row = {}
                for col, val in row.items():
                    if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                        formatted_row[col] = val.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(val) else val
                    else:
                        formatted_row[col] = val
                
                # Create a more structured content string
                content_parts = []
                for col, val in formatted_row.items():
                    if pd.notna(val):
                        content_parts.append(f"{col}: {val}")
                
                content = " | ".join(content_parts)
                
                # Add metadata for better retrieval
                metadata = {
                    "row_index": row.name,
                    "type": "data_row"
                }
                
                # Add any datetime values to metadata for time-based queries
                for col, val in formatted_row.items():
                    if 'time' in col.lower() or 'date' in col.lower():
                        if pd.notna(val):
                            metadata[f"{col}_timestamp"] = val
                
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
        
        return documents
    
    def _create_chain(self):
        try:
            documents = self._prepare_documents()
            
            # Adjust chunk size based on content
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,  # Smaller chunks for more precise retrieval
                chunk_overlap=100,
                length_function=len,
                separators=["\n", " | ", ", ", " "]
            )
            splits = text_splitter.split_documents(documents)
            
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(splits, embeddings)
            
            retriever = vectorstore.as_retriever(
                search_type="mmr",  # Use MMR for better diversity in results
                search_kwargs={
                    "k": 1000,  # Retrieve more documents for better context
                    "fetch_k": 20  # Fetch more documents before filtering
                }
            )
            
            template = """You are a precise data analyst. Answer the question based on the provided context.
            Follow these rules:
            1. Use exact values from the data
            2. For numerical questions, show the calculation
            3. For time-based questions, use exact timestamps
            4. If aggregating data, mention the number of records considered
            5. If the answer cannot be found in the context, say "I cannot find this information in the data"
            
            Context: {context}
            Question: {question}
            
            Answer:"""
            
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(
                    temperature=0,  # Set to 0 for more precise answers
                    model_name="gpt-3.5-turbo-16k"  # Use 16k model for longer context
                ),
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={
                    "prompt": prompt,
                    "verbose": True
                },
                return_source_documents=True  # This helps with debugging
            )
            
            return chain
            
        except Exception as e:
            print(f"Error creating chain: {str(e)}")
            return None
    
    def ask(self, question: str) -> str:
        if self.chain is None:
            return "System not initialized properly."
        try:
            result = self.chain({"query": question})
            answer = result['result']
            
            # Add source information for verification
            if 'source_documents' in result:
                sources = len(result['source_documents'])
                answer += f"\n\n(Based on {sources} relevant records)"
            
            return answer
        except Exception as e:
            return f"Error processing question: {str(e)}"

def main():
    print("Initializing system (this may take a few minutes for large datasets)...")
    rag = RAGSystem('atharva-prep-dataset - master-data.csv')
    print("\nSystem ready. Type 'exit' to quit.")
    print("You can ask questions about:")
    print("- Specific values (e.g., 'What was the total weight at timestamp X?')")
    print("- Aggregations (e.g., 'What is the total weight for ingredient Y?')")
    print("- Time-based queries (e.g., 'What ingredients were recorded on date Z?')")
    
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() == 'exit':
            break
        if question:
            print("\nProcessing...")
            answer = rag.ask(question)
            print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main() 