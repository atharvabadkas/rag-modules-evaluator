from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import numpy as np

class TextModel:
    def __init__(self):
        # Initialize the model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize knowledge base with some example QA pairs
        self.knowledge_base = {
            "What are the three laws of motion?": 
                "Newton's three laws of motion are: 1) An object at rest stays at rest, and an object in motion stays in motion unless acted upon by a force. 2) Force equals mass times acceleration (F=ma). 3) For every action, there is an equal and opposite reaction.",
            "What is the capital of France?": 
                "The capital of France is Paris.",
            "What is the speed of light?": 
                "The speed of light in vacuum is approximately 299,792,458 meters per second.",
        }
        
        # Pre-compute embeddings for all questions in knowledge base
        self.question_embeddings = {
            q: self.model.encode(q) for q in self.knowledge_base.keys()
        }
    
    def find_most_similar_question(self, question: str) -> Tuple[str, float]:
        """
        Find the most similar question in the knowledge base using cosine similarity
        """
        # Encode the input question
        query_embedding = self.model.encode(question)
        
        # Calculate similarities with all questions in knowledge base
        similarities = {}
        for q, emb in self.question_embeddings.items():
            similarity = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            similarities[q] = similarity
        
        # Find the most similar question
        most_similar = max(similarities.items(), key=lambda x: x[1])
        return most_similar
    
    def ask_question(self, question: str, similarity_threshold: float = 0.7) -> str:
        try:
            # Find most similar question in knowledge base
            similar_question, similarity = self.find_most_similar_question(question)
            
            # If similarity is above threshold, return the answer
            if similarity >= similarity_threshold:
                return self.knowledge_base[similar_question]
            else:
                return f"I'm sorry, I don't have enough information to answer that question confidently. (Similarity: {similarity:.2f})"
                
        except Exception as e:
            return f"Error processing question: {str(e)}"

    def add_to_knowledge_base(self, question: str, answer: str):
        """
        Add a new question-answer pair to the knowledge base
        """
        self.knowledge_base[question] = answer
        self.question_embeddings[question] = self.model.encode(question)

if __name__ == "__main__":
    # Initialize the model
    model = TextModel()
    
    # Test with various questions
    test_questions = [
        "What are Newton's laws?",
        "Tell me about the speed of light",
        "What's the capital city of France?",
        "What is the meaning of life?",  # This should return low similarity
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        response = model.ask_question(question)
        print(f"Answer: {response}")