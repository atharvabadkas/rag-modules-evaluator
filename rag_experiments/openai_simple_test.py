from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import pandas as pd
import os
from typing import List, Dict
import numpy as np
from datetime import datetime

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

class DataAnalyzer:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.chain = None
        self.column_descriptions = {}
        self.conversation_history = []
        
    def load_and_analyze_data(self):
        """
        Load the dataset and perform initial analysis
        """
        try:
            self.df = pd.read_csv(self.csv_path)
            # Convert timestamp columns to datetime if they exist
            for col in self.df.columns:
                if 'time' in col.lower() or 'date' in col.lower():
                    try:
                        self.df[col] = pd.to_datetime(self.df[col])
                    except:
                        pass  # If conversion fails, keep original format
                        
            print(f"Successfully loaded dataset with {len(self.df)} records")
            
            self.column_descriptions = self._analyze_columns()
            self.chain = self._create_rag_chain()
            return True
        except Exception as e:
            print(f"Error initializing system: {str(e)}")
            return False
    
    def _analyze_columns(self) -> Dict:
        """
        Analyze dataset columns and generate comprehensive descriptions
        """
        descriptions = {}
        for column in self.df.columns:
            col_data = self.df[column]
            col_type = col_data.dtype
            unique_count = col_data.nunique()
            null_count = col_data.isnull().sum()
            
            base_info = {
                'type': str(col_type),
                'unique_values': unique_count,
                'null_count': null_count,
                'total_count': len(col_data)
            }
            
            if pd.api.types.is_numeric_dtype(col_type):
                base_info.update({
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'mean': float(col_data.mean()),
                    'median': float(col_data.median()),
                    'std': float(col_data.std()),
                    'quartiles': col_data.quantile([0.25, 0.75]).to_dict()
                })
            elif pd.api.types.is_datetime64_any_dtype(col_type):
                base_info.update({
                    'min_date': col_data.min().strftime('%Y-%m-%d %H:%M:%S'),
                    'max_date': col_data.max().strftime('%Y-%m-%d %H:%M:%S'),
                    'date_range': str(col_data.max() - col_data.min())
                })
            else:
                if unique_count <= 10:  # For categorical with few unique values
                    base_info['value_counts'] = col_data.value_counts().to_dict()
                else:
                    base_info['top_values'] = col_data.value_counts().head(5).to_dict()
            
            descriptions[column] = base_info
        return descriptions

    def _get_statistical_summary(self, question: str) -> str:
        """
        Generate statistical summary based on the question
        """
        try:
            # Extract column names mentioned in the question
            mentioned_cols = [col for col in self.df.columns if col.lower() in question.lower()]
            
            if not mentioned_cols:
                return ""
                
            summaries = []
            for col in mentioned_cols:
                if pd.api.types.is_numeric_dtype(self.df[col].dtype):
                    stats = self.column_descriptions[col]
                    summary = f"\nStatistical Summary for {col}:"
                    summary += f"\n- Count: {stats['total_count']}"
                    summary += f"\n- Mean: {stats['mean']:.2f}"
                    summary += f"\n- Median: {stats['median']:.2f}"
                    summary += f"\n- Std Dev: {stats['std']:.2f}"
                    summary += f"\n- Min: {stats['min']:.2f}"
                    summary += f"\n- Max: {stats['max']:.2f}"
                    summary += f"\n- 25th percentile: {stats['quartiles'][0.25]:.2f}"
                    summary += f"\n- 75th percentile: {stats['quartiles'][0.75]:.2f}"
                    summaries.append(summary)
                    
            return "\n".join(summaries)
        except Exception as e:
            return f"Error generating statistical summary: {str(e)}"

    def prepare_data_for_rag(self) -> List[Document]:
        """
        Convert DataFrame into documents with dynamic field handling
        """
        documents = []
        # Create summary documents for each column
        for col, desc in self.column_descriptions.items():
            summary_text = f"Column {col} summary:\n"
            for key, value in desc.items():
                summary_text += f"{key}: {value}\n"
            documents.append(Document(
                page_content=summary_text,
                metadata={"type": "column_summary", "column": col}
            ))
        
        # Create documents for actual data records
        for idx, row in self.df.iterrows():
            text_parts = [f"Record {idx}:"]
            for col in self.df.columns:
                value = row[col]
                if pd.notna(value):
                    text_parts.append(f"{col}: {value}")
            
            text = " | ".join(text_parts)
            metadata = {col: row[col] for col in self.df.columns if pd.notna(row[col])}
            metadata["record_index"] = idx
            
            documents.append(Document(
                page_content=text,
                metadata=metadata
            ))
        return documents

    def _create_rag_chain(self):
        """
        Create an enhanced RAG chain with dataset-specific context
        """
        try:
            documents = self.prepare_data_for_rag()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            splits = text_splitter.split_documents(documents)
            
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(splits, embeddings)
            
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10}  # Increased for better coverage
            )
            
            # Enhanced prompt template with dataset context
            column_info = "\n".join([
                f"- {col}: {desc['type']} (unique: {desc['unique_values']}, null: {desc['null_count']})"
                for col, desc in self.column_descriptions.items()
            ])
            
            prompt_template = f"""You are an advanced data analysis assistant specialized in providing insights about datasets.
            
            Dataset Information:
            - Total Records: {len(self.df)}
            - Available Columns:
            {column_info}

            Context from the dataset:
            -------------------------
            {{context}}
            -------------------------

            Previous conversation:
            {self._format_conversation_history()}

            Important Instructions:
            1. For numerical calculations (mean, median, etc.), use ALL available data, not just the examples shown.
            2. When analyzing trends or patterns, consider the entire dataset.
            3. If asked about specific values or examples, use the context provided.
            4. Always mention the sample size or date range when relevant.
            5. If the information isn't available in the context, say so clearly.
            
            Question: {{question}}
            Answer: """
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(
                    temperature=0.1,
                    model_name="gpt-3.5-turbo"
                ),
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            print("Analysis system initialized successfully")
            return chain
            
        except Exception as e:
            print(f"Error creating analysis chain: {str(e)}")
            return None

    def _format_conversation_history(self) -> str:
        if not self.conversation_history:
            return "No previous conversation."
        
        history = []
        for i, (q, a) in enumerate(self.conversation_history[-3:], 1):
            history.extend([f"Q{i}: {q}", f"A{i}: {a}"])
        return "\n".join(history)

    def query(self, question: str) -> str:
        """
        Process a question and return the answer with comprehensive analysis
        """
        if self.chain is None:
            return "Error: System not properly initialized."
        
        try:
            print("\nProcessing query...")
            
            # Get statistical summary if relevant
            stats_summary = self._get_statistical_summary(question)
            
            # Get contextual answer
            result = self.chain({"query": question})
            answer = result['result']
            sources = result['source_documents']
            
            # Store in conversation history
            self.conversation_history.append((question, answer))
            
            # Format the response
            response = "\n" + "="*50
            response += f"\nAnswer: {answer}\n"
            
            # Add statistical summary if available
            if stats_summary:
                response += f"\nDetailed Statistics:{stats_summary}\n"
            
            response += "\nSupporting Data:"
            for i, doc in enumerate(sources[:3], 1):
                response += f"\n{i}. {doc.page_content}"
            response += "\n" + "="*50
            
            return response
            
        except Exception as e:
            return f"Error processing query: {str(e)}"

def interactive_session(csv_path: str):
    """
    Start an interactive analysis session
    """
    print("\nInitializing data analysis system...")
    analyzer = DataAnalyzer(csv_path)
    
    if not analyzer.load_and_analyze_data():
        print("Failed to initialize the system. Please check the error messages above.")
        return
    
    print("\nSystem is ready.")
    print("Type 'exit' to quit, 'help' for command list")
    
    while True:
        question = input("\nEnter your question: ").strip()
        
        if question.lower() == 'exit':
            print("\nEnding session.")
            break
            
        if question.lower() == 'help':
            print("\nAvailable commands:")
            print("- Type any question about the data")
            print("- 'help': Show this help message")
            print("- 'exit': End the session")
            continue
            
        if not question:
            print("Please enter a valid question.")
            continue
            
        answer = analyzer.query(question)
        print(answer)

if __name__ == "__main__":
    interactive_session("atharva-prep-dataset - master-data.csv")