from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
from scipy import stats
import json
import os

# Set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = "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"

class DatasetManager:
    def __init__(self):
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.dataset_metadata: Dict[str, Dict] = {}
        
    def add_dataset(self, name: str, filepath: str, dataset_type: str = "dynamic"):
        """
        Load a dataset and store its metadata
        dataset_type can be 'dynamic' (regularly updated) or 'static' (reference data)
        """
        try:
            df = pd.read_csv(filepath)
            
            # Convert date/time columns
            date_columns = []
            for col in df.columns:
                if any(term in col.lower() for term in ['date', 'time', 'timestamp']):
                    try:
                        df[col] = pd.to_datetime(df[col])
                        date_columns.append(col)
                    except:
                        pass
            
            self.datasets[name] = df
            
            # Store metadata
            self.dataset_metadata[name] = {
                'type': dataset_type,
                'filepath': filepath,
                'rows': len(df),
                'columns': list(df.columns),
                'date_columns': date_columns,
                'last_updated': datetime.now().isoformat(),
                'column_types': {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
            
            return True
        except Exception as e:
            print(f"Error loading dataset {name}: {str(e)}")
            return False

class AnalyticsEngine:
    """Handles complex calculations and analysis across datasets"""
    
    def __init__(self, dataset_manager: DatasetManager):
        self.dm = dataset_manager
    
    def time_series_analysis(self, dataset_name: str, column: str, 
                           time_column: str, grouping: str = 'D',
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> Dict:
        """
        Perform time series analysis on specified column
        grouping: 'D' for daily, 'W' for weekly, 'M' for monthly
        """
        try:
            df = self.dm.datasets[dataset_name]
            if start_date:
                df = df[df[time_column] >= start_date]
            if end_date:
                df = df[df[time_column] <= end_date]
                
            grouped = df.groupby(pd.Grouper(key=time_column, freq=grouping))[column]
            
            result = {
                'trend': grouped.mean().to_dict(),
                'total_by_period': grouped.sum().to_dict(),
                'growth_rate': grouped.mean().pct_change().mean(),
                'seasonality': self._check_seasonality(grouped.mean()),
                'statistics': {
                    'mean': float(grouped.mean().mean()),
                    'std': float(grouped.mean().std()),
                    'min': float(grouped.mean().min()),
                    'max': float(grouped.mean().max())
                }
            }
            return result
        except Exception as e:
            return {'error': str(e)}
    
    def _check_seasonality(self, series: pd.Series) -> Dict:
        """Check for seasonal patterns in time series"""
        try:
            acf = stats.acf(series.dropna(), nlags=len(series)//2)
            return {
                'has_seasonality': bool(np.any(acf[1:] > 0.7)),
                'strongest_period': int(np.argmax(acf[1:]) + 1) if np.any(acf[1:] > 0.7) else None
            }
        except:
            return {'has_seasonality': False, 'strongest_period': None}
    
    def cross_dataset_analysis(self, datasets: List[str], 
                             columns: Dict[str, str],
                             filters: Optional[Dict] = None) -> Dict:
        """
        Perform analysis across multiple datasets
        """
        results = {}
        try:
            # Apply filters and get relevant data from each dataset
            filtered_data = {}
            for dataset in datasets:
                df = self.dm.datasets[dataset].copy()
                if filters and dataset in filters:
                    for col, condition in filters[dataset].items():
                        df = df[df[col].isin(condition) if isinstance(condition, list) 
                              else df[col] == condition]
                filtered_data[dataset] = df
            
            # Calculate basic statistics for each dataset
            for dataset, df in filtered_data.items():
                if dataset in columns:
                    col = columns[dataset]
                    results[dataset] = {
                        'count': len(df),
                        'sum': float(df[col].sum()),
                        'mean': float(df[col].mean()),
                        'median': float(df[col].median()),
                        'std': float(df[col].std())
                    }
            
            # Calculate correlations if possible
            if len(datasets) == 2:
                try:
                    df1, df2 = filtered_data[datasets[0]], filtered_data[datasets[1]]
                    col1, col2 = columns[datasets[0]], columns[datasets[1]]
                    correlation = df1[col1].corr(df2[col2])
                    results['correlation'] = float(correlation)
                except:
                    results['correlation'] = None
                    
            return results
        except Exception as e:
            return {'error': str(e)}

class DataAnalyzer:
    def __init__(self):
        self.dataset_manager = DatasetManager()
        self.analytics_engine = AnalyticsEngine(self.dataset_manager)
        self.chain = None
        self.conversation_history = []
    
    def load_datasets(self, dataset_configs: List[Dict]) -> bool:
        """
        Load multiple datasets with their configurations
        dataset_configs: List of dicts with 'name', 'path', and 'type'
        """
        success = True
        for config in dataset_configs:
            if not self.dataset_manager.add_dataset(
                config['name'], config['path'], config.get('type', 'dynamic')):
                success = False
        
        if success:
            self.chain = self._create_rag_chain()
        return success
    
    def prepare_data_for_rag(self) -> List[Document]:
        """
        Convert all datasets into documents with enhanced context
        """
        documents = []
        
        # Add dataset metadata documents
        for name, metadata in self.dataset_manager.dataset_metadata.items():
            meta_doc = Document(
                page_content=f"Dataset {name} information:\n" + 
                            json.dumps(metadata, indent=2),
                metadata={"type": "dataset_metadata", "dataset": name}
            )
            documents.append(meta_doc)
        
        # Add data documents from each dataset
        for name, df in self.dataset_manager.datasets.items():
            # Add column summaries
            for col in df.columns:
                summary = self._generate_column_summary(df, col)
                doc = Document(
                    page_content=f"Summary of column {col} in dataset {name}:\n{summary}",
                    metadata={"type": "column_summary", "dataset": name, "column": col}
                )
                documents.append(doc)
            
            # Add actual data records
            for idx, row in df.iterrows():
                text_parts = [f"Record {idx} from dataset {name}:"]
                for col in df.columns:
                    if pd.notna(row[col]):
                        text_parts.append(f"{col}: {row[col]}")
                
                doc = Document(
                    page_content=" | ".join(text_parts),
                    metadata={
                        "type": "record",
                        "dataset": name,
                        "record_index": idx,
                        **{col: row[col] for col in df.columns if pd.notna(row[col])}
                    }
                )
                documents.append(doc)
        
        return documents
    
    def _generate_column_summary(self, df: pd.DataFrame, column: str) -> str:
        """Generate comprehensive summary for a column"""
        try:
            summary = []
            data = df[column]
            
            # Basic info
            summary.append(f"Type: {data.dtype}")
            summary.append(f"Total values: {len(data)}")
            summary.append(f"Unique values: {data.nunique()}")
            summary.append(f"Null values: {data.isnull().sum()}")
            
            # Type-specific analysis
            if pd.api.types.is_numeric_dtype(data.dtype):
                summary.extend([
                    f"Mean: {data.mean():.2f}",
                    f"Median: {data.median():.2f}",
                    f"Std: {data.std():.2f}",
                    f"Min: {data.min():.2f}",
                    f"Max: {data.max():.2f}"
                ])
            elif pd.api.types.is_datetime64_any_dtype(data.dtype):
                summary.extend([
                    f"Date range: {data.min()} to {data.max()}",
                    f"Unique dates: {data.dt.date.nunique()}"
                ])
            else:
                if data.nunique() <= 10:
                    summary.append("Value counts:")
                    for val, count in data.value_counts().items():
                        summary.append(f"  {val}: {count}")
                else:
                    summary.append("Top 5 values:")
                    for val, count in data.value_counts().head().items():
                        summary.append(f"  {val}: {count}")
            
            return "\n".join(summary)
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def _create_rag_chain(self):
        """
        Create enhanced RAG chain with multi-dataset support
        """
        try:
            documents = self.prepare_data_for_rag()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=3000,
                chunk_overlap=200,
                length_function=len
            )
            splits = text_splitter.split_documents(documents)
            
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(splits, embeddings)
            
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 1200}  # Increased for better multi-dataset coverage
            )
            
            # Create dataset summaries
            dataset_summaries = []
            for name, metadata in self.dataset_manager.dataset_metadata.items():
                summary = f"- {name} ({metadata['type']}): {metadata['rows']} records"
                if metadata['date_columns']:
                    summary += f", date columns: {', '.join(metadata['date_columns'])}"
                dataset_summaries.append(summary)
            
            prompt_template = f"""You are an advanced data analysis assistant specialized in multi-dataset analysis.

            Available Datasets:
            {chr(10).join(dataset_summaries)}

            Context from the datasets:
            -------------------------
            {{context}}
            -------------------------

            Previous conversation:
            {self._format_conversation_history()}

            Instructions for Analysis:
            1. For calculations, use complete data from relevant datasets
            2. For time-series analysis, consider the full date range
            3. For cross-dataset analysis, verify data compatibility
            4. Always specify which dataset and time period was used
            5. For aggregations, mention the grouping criteria
            6. If comparing datasets, note any assumptions made
            7. For predictions, explain the basis of the forecast
            
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
            
            print("Multi-dataset analysis system initialized")
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
        Process a question with enhanced multi-dataset analysis
        """
        if self.chain is None:
            return "Error: System not properly initialized."
        
        try:
            print("\nAnalyzing query...")
            
            # Determine if this is a complex analysis query
            is_time_series = any(term in question.lower() 
                               for term in ['trend', 'over time', 'pattern', 'monthly', 'weekly'])
            is_cross_dataset = sum(dataset in question 
                                 for dataset in self.dataset_manager.datasets.keys()) > 1
            
            # Get base RAG response
            result = self.chain({"query": question})
            answer = result['result']
            sources = result['source_documents']
            
            # Add additional analysis if needed
            additional_analysis = []
            
            if is_time_series:
                # Identify relevant dataset and column for time series analysis
                # This is a simplified example - you'd need more sophisticated parsing
                for dataset, metadata in self.dataset_manager.dataset_metadata.items():
                    if dataset in question and metadata['date_columns']:
                        for col in metadata['columns']:
                            if col in question and pd.api.types.is_numeric_dtype(
                                self.dataset_manager.datasets[dataset][col].dtype):
                                analysis = self.analytics_engine.time_series_analysis(
                                    dataset, col, metadata['date_columns'][0])
                                additional_analysis.append(
                                    f"\nTime Series Analysis for {col} in {dataset}:"
                                    f"\n- Growth Rate: {analysis['growth_rate']:.2%}"
                                    f"\n- Seasonality: {'Detected' if analysis['seasonality']['has_seasonality'] else 'Not detected'}"
                                )
            
            # Store in conversation history
            self.conversation_history.append((question, answer))
            
            # Format the response
            response = "\n" + "="*50
            response += f"\nAnswer: {answer}\n"
            
            # if additional_analysis:
            #     response += "\nAdditional Analysis:"
            #     response += "\n".join(additional_analysis)
            
            # response += "\nSupporting Data:"
            # for i, doc in enumerate(sources[:3], 1):
            #     response += f"\n{i}. {doc.page_content}"
            # response += "\n" + "="*50
            
            return response
            
        except Exception as e:
            return f"Error processing query: {str(e)}"

def interactive_session(dataset_configs: List[Dict]):
    """
    Start an interactive session with multiple datasets
    """
    print("\nInitializing multi-dataset analysis system...")
    analyzer = DataAnalyzer()
    
    if not analyzer.load_datasets(dataset_configs):
        print("Failed to initialize the system. Please check the error messages above.")
        return
    
    print("\nAnalysis system is ready.")
    print("Type 'exit' to quit, 'help' for command list")
    
    while True:
        question = input("\nEnter your question: ").strip()
        
        if question.lower() == 'exit':
            print("\nEnding analysis session.")
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
    # Example configuration for multiple datasets
    dataset_configs = [
        {
            'name': 'preparation',
            'path': 'atharva-prep-dataset - master-data.csv',
            'type': 'dynamic'
        }
        # Add other datasets as needed
    ]
    
    interactive_session(dataset_configs) 