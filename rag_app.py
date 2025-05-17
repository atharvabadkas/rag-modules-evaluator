import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

class RAGApp:
    def __init__(self, csv_path: str):
        """Initialize RAG application"""
        load_dotenv()
        self.api_key = os.getenv('OPENAI_API_KEY')
        
        print("Loading dataset...")
        self.df = pd.read_csv(csv_path)
        self.preprocess_data()
        self.create_documents()
        self.setup_rag_components()

    def preprocess_data(self):
        """Preprocess the dataset"""
        self.df.columns = self.df.columns.str.lower()
        
        # Process each column appropriately
        for col in self.df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                except:
                    pass
            else:
                try:
                    self.df[col] = pd.to_numeric(self.df[col], errors='ignore')
                except:
                    pass

    def create_documents(self):
        """Create documents for RAG with better chunking"""
        print("Creating documents...")
        self.documents = []
        
        # Add dataset overview
        overview = (
            f"Dataset Overview:\n"
            f"Total Records: {len(self.df):,}\n"
            f"Columns: {', '.join(self.df.columns)}\n"
        )
        self.documents.append(Document(
            page_content=overview,
            metadata={'type': 'overview'}
        ))
        
        # Process each row with detailed formatting
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing records"):
            # Format each field properly
            fields = []
            metadata = {'row_id': idx}
            
            for col, val in row.items():
                if pd.notna(val):
                    if isinstance(val, pd.Timestamp):
                        formatted_val = val.strftime('%Y-%m-%d %H:%M:%S')
                    elif isinstance(val, (int, float)):
                        formatted_val = f"{val:,}"
                    else:
                        formatted_val = str(val)
                    
                    fields.append(f"{col}: {formatted_val}")
                    metadata[col] = formatted_val
            
            # Create detailed document
            content = (
                f"Record {idx + 1}:\n" + 
                "\n".join(fields)
            )
            
            self.documents.append(Document(
                page_content=content,
                metadata=metadata
            ))

    def setup_rag_components(self):
        """Setup RAG components with better retrieval"""
        print("Setting up RAG components...")
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.api_key
        )
        
        # Create vector store
        self.vector_store = FAISS.from_documents(
            self.documents,
            self.embeddings
        )
        
        # Setup LLM
        self.llm = ChatOpenAI(
            temperature=0,
            model_name='gpt-3.5-turbo-16k',
            openai_api_key=self.api_key
        )

        # Setup memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        # Create retrieval chain with memory
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 10}
            ),
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={
                'prompt': ChatPromptTemplate.from_messages([
                    ("system", """You are a data analyst assistant. Answer questions about the dataset accurately.
                    
                    Guidelines:
                    1. Use ONLY information from the provided context
                    2. For numerical questions, use exact numbers from the context
                    3. For timestamps, be precise about dates and times
                    4. If information is not in the context, say so
                    5. Always cite the record numbers you used
                    
                    Context: {context}"""),
                    ("human", "{question}")
                ])
            }
        )
        
        print("RAG system ready!")

    def ask(self, question: str) -> str:
        """Process question using RAG"""
        try:
            # Get response with source documents
            result = self.chain({"question": question})
            
            # Extract answer and sources
            answer = result['answer']
            sources = result.get('source_documents', [])
            
            # Format response with sources
            response = answer.strip()
            if sources:
                source_texts = []
                for doc in sources:
                    if doc.metadata.get('type') != 'overview':
                        record_num = f"Record {doc.metadata.get('row_id', 'Unknown') + 1}"
                        # Add timestamp if available
                        timestamp_cols = [col for col in doc.metadata.keys() 
                                       if 'date' in col.lower() or 'time' in col.lower()]
                        if timestamp_cols:
                            record_num += f" ({doc.metadata[timestamp_cols[0]]})"
                        source_texts.append(record_num)
                
                if source_texts:
                    response += f"\n\nSources: {', '.join(source_texts)}"
            
            return response
            
        except Exception as e:
            return f"Error: {str(e)}"

# Interactive usage
if __name__ == "__main__":
    rag = RAGApp('atharva-prep-dataset - master-data.csv')
    
    print("\n=== Interactive Q&A Session ===")
    print("Type 'exit' to end the conversation\n")
    
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() == 'exit':
            break
            
        print("\nSearching...\n")
        answer = rag.ask(question)
        print(f"Answer: {answer}\n")