import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any
import json
import pickle
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_time=False)]
)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, csv_path: str, cache_dir: str = "cache"):
        """Handle data loading, preprocessing, and caching"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load and preprocess data
        logger.info("Loading dataset...")
        self.df = pd.read_csv(csv_path)
        self.preprocess_data()
        self.compute_statistics()

    def preprocess_data(self):
        """Preprocess the dataframe"""
        # Normalize column names
        self.df.columns = self.df.columns.str.lower()
        
        # Convert numeric columns
        for col in self.df.select_dtypes(include=[np.number]).columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute and cache dataset statistics"""
        try:
            cached_stats = self._load_cache("statistics")
            if cached_stats:
                self.statistics = cached_stats
                return self.statistics

            self.statistics = {
                'columns': list(self.df.columns),
                'numeric_stats': self._compute_numeric_stats(),
                'top_values': self._compute_top_values(),
                'row_count': len(self.df)
            }
            
            self._save_cache(self.statistics, "statistics")
            return self.statistics
            
        except Exception as e:
            logger.error(f"Error computing statistics: {e}")
            return {}

    def _compute_numeric_stats(self) -> Dict[str, Dict[str, float]]:
        """Compute statistics for numeric columns"""
        stats = {}
        for col in self.df.select_dtypes(include=[np.number]).columns:
            stats[col] = {
                'mean': float(self.df[col].mean()),
                'median': float(self.df[col].median()),
                'max': float(self.df[col].max()),
                'min': float(self.df[col].min()),
                'std': float(self.df[col].std())
            }
        return stats

    def _compute_top_values(self, n: int = 10) -> Dict[str, list]:
        """Compute top values for relevant columns"""
        top_values = {}
        for col in self.df.columns:
            if self.df[col].nunique() < 100:  # Only for columns with reasonable cardinality
                value_counts = self.df[col].value_counts().head(n)
                top_values[col] = value_counts.to_dict()
        return top_values

    def _save_cache(self, data: Dict, filename: str):
        """Save data to cache"""
        try:
            cache_path = self.cache_dir / f"{filename}.pkl"
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Pickle save failed: {e}, trying JSON...")
            try:
                with open(self.cache_dir / f"{filename}.json", 'w') as f:
                    json.dump(data, f)
            except Exception as e:
                logger.error(f"Cache save failed: {e}")

    def _load_cache(self, filename: str) -> Dict:
        """Load data from cache"""
        for ext in ['.pkl', '.json']:
            path = self.cache_dir / f"{filename}{ext}"
            if path.exists():
                try:
                    if ext == '.pkl':
                        with open(path, 'rb') as f:
                            return pickle.load(f)
                    else:
                        with open(path, 'r') as f:
                            return json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
        return None 