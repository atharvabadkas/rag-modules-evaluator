import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
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
        self.load_and_process_data(csv_path)
        self.setup_components()
        self.create_vector_store()
        self.create_timestamp_index()

    def load_and_process_data(self, csv_path: str):
        """Load and process the dataset"""
        # Load data
        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.lower()
        
        # Identify and convert timestamp columns
        self.timestamp_columns = []
        for col in self.df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                    self.timestamp_columns.append(col)
                except:
                    pass
            else:
                # Try converting to numeric for non-date columns
                try:
                    self.df[col] = pd.to_numeric(self.df[col], errors='ignore')
                except:
                    pass

        print("Processing dataset...")
        self.documents = []
        self._process_data_to_documents()

    def create_timestamp_index(self):
        """Create indices for timestamp-based searching"""
        self.timestamp_index = {}
        for col in self.timestamp_columns:
            # Create a dictionary mapping timestamps to row indices
            self.timestamp_index[col] = {
                str(ts): idx for idx, ts in enumerate(self.df[col])
                if pd.notna(ts)
            }

    def _process_data_to_documents(self):
        """Process data into documents with better timestamp handling"""
        # Add dataset overview
        stats = self._generate_dataset_stats()
        self.documents.append(Document(
            page_content=stats,
            metadata={'type': 'statistics'}
        ))
        
        # Process records in chunks
        chunk_size = 1000
        for start_idx in tqdm(range(0, len(self.df), chunk_size), desc="Processing records"):
            chunk = self.df.iloc[start_idx:start_idx + chunk_size]
            for idx, row in chunk.iterrows():
                details = []
                metadata = {'row_id': idx, 'record_num': idx + 1}
                
                for col, val in row.items():
                    if pd.notna(val):
                        if col in self.timestamp_columns:
                            # Format timestamp consistently
                            formatted_val = pd.to_datetime(val).strftime('%Y-%m-%d %H:%M:%S')
                            metadata[col] = formatted_val
                        elif isinstance(val, (int, float)):
                            formatted_val = f"{val:,}"
                            metadata[col] = str(val)
                        else:
                            formatted_val = str(val)
                            metadata[col] = str(val)
                        details.append(f"{col}: {formatted_val}")
                
                content = (
                    f"Record {idx + 1}:\n" + 
                    "\n".join(details)
                )
                self.documents.append(Document(
                    page_content=content,
                    metadata=metadata
                ))

    def _generate_dataset_stats(self) -> str:
        """Generate comprehensive dataset statistics"""
        stats_parts = ["Dataset Statistics:"]
        
        # Basic info
        stats_parts.append(f"Total Records: {len(self.df):,}")
        stats_parts.append(f"Columns: {', '.join(self.df.columns)}")
        
        # Numeric column statistics
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            stats_parts.append("\nNumerical Column Statistics:")
            for col in numeric_cols:
                stats = self.df[col].describe()
                stats_parts.append(f"\n{col}:")
                stats_parts.append(f"  Min: {stats['min']:,.2f}")
                stats_parts.append(f"  Max: {stats['max']:,.2f}")
                stats_parts.append(f"  Mean: {stats['mean']:,.2f}")
                stats_parts.append(f"  Total: {self.df[col].sum():,.2f}")

        # Categorical column statistics
        cat_cols = self.df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            stats_parts.append("\nCategorical Column Statistics:")
            for col in cat_cols:
                unique_vals = self.df[col].nunique()
                stats_parts.append(f"\n{col}:")
                stats_parts.append(f"  Unique Values: {unique_vals:,}")
                if unique_vals < 10:  # Only show value counts for columns with few unique values
                    value_counts = self.df[col].value_counts()
                    for val, count in value_counts.items():
                        stats_parts.append(f"  - {val}: {count:,}")

        return "\n".join(stats_parts)

    def setup_components(self):
        """Setup LangChain components"""
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.api_key
        )
        
        self.llm = ChatOpenAI(
            temperature=0,
            model_name='gpt-3.5-turbo-16k',
            openai_api_key=self.api_key
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )

    def create_vector_store(self):
        """Create vector store and setup retrieval chain"""
        print("Creating vector store...")
        self.vector_store = FAISS.from_documents(
            documents=self.documents,
            embedding=self.embeddings
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 30}  # Increased for better coverage
            ),
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={
                'prompt': ChatPromptTemplate.from_messages([
                    ("system", """You are a data analyst assistant with comprehensive access to a large dataset. 
                    Your task is to provide detailed, accurate answers based on the data.

                    Important Guidelines:
                    1. Always check the dataset statistics first for an overview
                    2. For numerical questions:
                       - Perform precise calculations
                       - Include totals, averages, and trends where relevant
                       - Double-check your math
                    3. For listing or ranking:
                       - Consider ALL relevant records
                       - Sort appropriately (e.g., highest to lowest)
                       - Provide complete lists unless specifically limited
                    4. When comparing:
                       - Use exact numbers from the data
                       - Consider all relevant factors
                    5. Always cite your sources using record numbers
                    6. If you're unsure or need more context, say so
                    
                    Context: {context}"""),
                    ("human", "{question}")
                ])
            }
        )
        print("Ready for conversation!")

    def ask(self, question: str) -> str:
        """Enhanced question processing with better timestamp handling"""
        try:
            # Check if this is a timestamp-specific query
            if any(term in question.lower() for term in ['time', 'date', 'when']):
                print("Detected time-based query, searching for matching timestamps...")
                relevant_docs = self._get_timestamp_relevant_docs(question)
                
                if relevant_docs:
                    print(f"Found {len(relevant_docs)} matching records")
                    # Add these documents to the context first
                    context = "\n\n".join(doc.page_content for doc in relevant_docs)
                    
                    # Get additional context from vector search
                    vector_docs = self.vector_store.similarity_search(question, k=5)
                    additional_context = "\n\n".join(doc.page_content for doc in vector_docs)
                    
                    # Combine contexts
                    full_context = f"{context}\n\n{additional_context}"
                else:
                    print("No exact timestamp matches found, using vector search...")
                    # Fallback to regular vector search
                    docs = self.vector_store.similarity_search(question, k=30)
                    full_context = "\n\n".join(doc.page_content for doc in docs)
            else:
                # Regular vector search for non-timestamp queries
                docs = self.vector_store.similarity_search(question, k=30)
                full_context = "\n\n".join(doc.page_content for doc in docs)

            # Get response using the chain
            result = self.chain({
                "question": question,
                "chat_history": self.memory.chat_memory.messages,
                "context": full_context
            })
            
            answer = result['answer']
            sources = result.get('source_documents', [])
            
            # Format response with timestamp information if available
            response = answer.strip()
            if sources:
                source_texts = []
                for s in sources:
                    if s.metadata.get('type') != 'statistics':
                        record_num = s.metadata.get('record_num', 'Unknown')
                        timestamp = next((s.metadata.get(col) for col in self.timestamp_columns 
                                       if col in s.metadata), None)
                        source_text = f"Record {record_num}"
                        if timestamp:
                            source_text += f" ({timestamp})"
                        source_texts.append(source_text)
                
                if source_texts:
                    response += f"\n\nSources: {', '.join(source_texts)}"
            
            return response
            
        except Exception as e:
            return f"Error: {str(e)}"

    def _get_timestamp_relevant_docs(self, question: str) -> list:
        """Find documents relevant to a timestamp query"""
        relevant_docs = []
        
        try:
            # Parse the question to extract date and time components
            question_parts = question.lower().split()
            
            for col in self.timestamp_columns:
                # Get all timestamps from the dataset
                timestamps = pd.to_datetime(self.df[col].dropna())
                
                for idx, ts in timestamps.items():
                    # Convert timestamp to different formats for comparison
                    ts_formats = {
                        'full': ts.strftime('%Y-%m-%d %H:%M:%S'),
                        'date': ts.strftime('%Y-%m-%d'),
                        'time': ts.strftime('%H:%M:%S'),
                        'components': {
                            'year': str(ts.year),
                            'month': str(ts.month),
                            'day': str(ts.day),
                            'hour': str(ts.hour),
                            'minute': str(ts.minute),
                            'second': str(ts.second)
                        }
                    }
                    
                    # Check if the timestamp matches the question
                    if self._check_timestamp_match(ts_formats, question_parts):
                        doc = next((doc for doc in self.documents 
                                  if doc.metadata.get('row_id') == idx), None)
                        if doc and doc not in relevant_docs:
                            relevant_docs.append(doc)
                            print(f"Found matching record for timestamp: {ts_formats['full']}")
        
        except Exception as e:
            print(f"Error in timestamp matching: {e}")
        
        return relevant_docs

    def _check_timestamp_match(self, ts_formats: dict, question_parts: list) -> bool:
        """Enhanced timestamp matching logic"""
        # Check full timestamp
        if ts_formats['full'] in ' '.join(question_parts):
            return True
            
        # Check date and time separately
        if ts_formats['date'] in ' '.join(question_parts):
            return True
            
        if ts_formats['time'] in ' '.join(question_parts):
            return True
        
        # Check individual components
        components = ts_formats['components']
        matches = 0
        required_matches = 0
        
        # Count how many components are mentioned in the question
        for part in question_parts:
            if part.isdigit():
                required_matches += 1
                if any(part == val for val in components.values()):
                    matches += 1
        
        # If all mentioned components match
        return required_matches > 0 and matches == required_matches

# Interactive usage
if __name__ == "__main__":
    rag = RAGApp('atharva-prep-dataset - master-data.csv')
    
    print("\n=== Interactive Q&A Session ===")
    print("Type 'exit' to end the conversation\n")
    
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() == 'exit':
            break
            
        print("\nThinking...\n")
        answer = rag.ask(question)
        print(f"Answer: {answer}") 