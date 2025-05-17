import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Any, List, Dict

class DataQuerySystem:
    def __init__(self, csv_path: str):
        """
        Initialize the query system with a CSV file
        
        Args:
            csv_path: Path to the CSV file
        """
        # Load the data
        self.df = pd.read_csv(csv_path)
        
        # Store basic information about the dataset
        self.columns = list(self.df.columns)
        self.num_rows = len(self.df)
        
        # Initialize the sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create a dictionary of common query patterns and their functions
        self.query_patterns = {
            "average": self._get_average,
            "mean": self._get_average,
            "maximum": self._get_maximum,
            "max": self._get_maximum,
            "minimum": self._get_minimum,
            "min": self._get_minimum,
            "count": self._get_count,
            "sum": self._get_sum,
            "show": self._show_data,
            "describe": self._describe_column
        }

    def _get_average(self, column: str) -> float:
        """Calculate average of a column"""
        if column in self.columns:
            try:
                return self.df[column].mean()
            except:
                return f"Cannot calculate average for column {column}"
        return f"Column {column} not found"

    def _get_maximum(self, column: str) -> Any:
        """Get maximum value in a column"""
        if column in self.columns:
            return self.df[column].max()
        return f"Column {column} not found"

    def _get_minimum(self, column: str) -> Any:
        """Get minimum value in a column"""
        if column in self.columns:
            return self.df[column].min()
        return f"Column {column} not found"

    def _get_count(self, column: str) -> int:
        """Get count of unique values in a column"""
        if column in self.columns:
            return self.df[column].nunique()
        return f"Column {column} not found"

    def _get_sum(self, column: str) -> float:
        """Calculate sum of a column"""
        if column in self.columns:
            try:
                return self.df[column].sum()
            except:
                return f"Cannot calculate sum for column {column}"
        return f"Column {column} not found"

    def _show_data(self, column: str, n: int = 5) -> pd.DataFrame:
        """Show first n rows of data"""
        if column == "all":
            return self.df.head(n)
        elif column in self.columns:
            return self.df[column].head(n)
        return f"Column {column} not found"

    def _describe_column(self, column: str) -> str:
        """Get statistical description of a column"""
        if column in self.columns:
            return str(self.df[column].describe())
        return f"Column {column} not found"

    def get_dataset_info(self) -> str:
        """Get basic information about the dataset"""
        info = f"""
        Dataset Summary:
        - Number of rows: {self.num_rows}
        - Number of columns: {len(self.columns)}
        - Columns: {', '.join(self.columns)}
        """
        return info

    def query(self, question: str) -> Any:
        """
        Process a natural language query about the dataset
        
        Args:
            question: Natural language question about the data
            
        Returns:
            Answer to the query
        """
        question = question.lower()
        
        # First, check if it's a request for dataset info
        if "what columns" in question or "show columns" in question or "dataset info" in question:
            return self.get_dataset_info()
        
        # Check for each query pattern
        for pattern, func in self.query_patterns.items():
            if pattern in question:
                # Try to identify the column name from the question
                # This is a simple approach - you might want to make this more sophisticated
                for col in self.columns:
                    if col.lower() in question.lower():
                        return func(col)
                
                # If looking for all columns
                if "all" in question or "data" in question:
                    return func("all")
                    
                return "Could not identify which column to analyze"
        
        return "I'm sorry, I don't understand that query. Try asking about average, maximum, minimum, count, or sum of a specific column."

# Example usage
if __name__ == "__main__":
    # Initialize the system with your CSV file
    query_system = DataQuerySystem("atharva-prep-dataset - master-data.csv")
    
    # Example queries
    example_questions = [
        "What columns are in the dataset?",
        "Show me the first 5 rows of data",
        "What is the maximum for from ingredient sku column",
        "What is the minimum for from ingredient sku column",
        "What is the count for from ingredient sku column",
        "What is the sum for from ingredient sku column",
        "What is the average for from ingredient sku column",
    ]
    
    for question in example_questions:
        print(f"\nQuestion: {question}")
        print(f"Answer: {query_system.query(question)}")