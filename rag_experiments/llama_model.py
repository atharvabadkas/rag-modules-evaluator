from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Initialize Ollama with your chosen model
llm = OllamaLLM(model="llama2")

# Create a prompt template
prompt = PromptTemplate(
    input_variables=["question"],
    template="Question: {question}\nAnswer:"
)

# Create a chain using the new pipe syntax
chain = prompt | llm | StrOutputParser()

# Run the chain
response = chain.invoke({"question": "can any model be trained on 3 to 4 images dataset for each catagory and having atleast 50 catagories, few catagories have few similarities"})
print(response)
