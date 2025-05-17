from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

def create_qa_system():
        # 1. Load your CSV document
    loader = CSVLoader(
    file_path='./atharva-prep-dataset - master-data.csv',
    encoding='utf-8'
        )
    documents = loader.load()
    # 2. Split documents into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)

    # 3. Create embeddings using Ollama
    embeddings = OllamaEmbeddings(model="llama2", base_url="http://localhost:11434")
    vectorstore_db = Chroma.from_documents(texts, embeddings)

    # 4. Create a retrieval-based QA chain using llama-2
    llm = Ollama(model="llama2", base_url="http://localhost:11434")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_db.as_retriever()
    )

    return qa_chain

# Create the QA system
qa = create_qa_system()

# Ask questions
while True:
    question = input("\nWhat is the ingredient id for Khobra? ")
    if question.lower() == 'quit':
        break
    
    answer = qa.run(question)
    print("\nAnswer:", answer)