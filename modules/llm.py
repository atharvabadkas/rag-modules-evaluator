from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from typing import List, Dict
import logging
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_time=False)]
)
logger = logging.getLogger(__name__)

class LLM:
    def __init__(self, api_key: str):
        self.chat = ChatOpenAI(
            temperature=0,
            model_name='gpt-3.5-turbo-16k',
            openai_api_key=api_key,
            max_tokens=4000
        )
        self._setup_prompt()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

    def _setup_prompt(self):
        """Setup the prompt template"""
        template = """
        You are a data analyst assistant. Answer the following question using the provided data.
        Keep your answers focused and precise.
        
        Question: {question}

        Instructions:
        1. Use statistics when available
        2. Provide specific values with sources
        3. Format numbers clearly
        4. Be concise and accurate

        Answer:
        """
        
        self.prompt = PromptTemplate(
            input_variables=["question"],
            template=template
        )

    def setup_chain(self, retriever):
        """Setup the conversational chain with token limits"""
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.chat,
            retriever=retriever,
            max_tokens_limit=4000,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": self.prompt}
        )

    def get_response(self, question: str, context: List[str], stats: Dict, columns: List[str]) -> str:
        """Get response from LLM"""
        try:
            # Format the question with context
            enhanced_question = f"""
            Context Information:
            Statistics: {stats if stats else 'Not required for this query'}
            
            {question}
            """
            
            # Use the chain to get response
            result = self.chain.invoke({"question": enhanced_question})
            
            # Extract answer and sources
            answer = result["answer"]
            sources = []
            if "source_documents" in result:
                for doc in result["source_documents"][:3]:
                    sources.append(doc.page_content)
            
            # Format the final response
            final_response = answer
            if sources:
                final_response += "\n\nSources:\n" + "\n\n".join(sources)
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            return f"Error: {str(e)}" 