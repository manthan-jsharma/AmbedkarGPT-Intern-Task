import sys
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate                   
from langchain_core.runnables import RunnablePassthrough              
from langchain_core.output_parsers import StrOutputParser


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistral"
PERSIST_DIR = "./chroma_db"
DATA_FILE = "speech.txt"

def build_rag_chain():
    
    print("Loading document...")
    loader = TextLoader(DATA_FILE, encoding="utf-8")
    docs = loader.load()

    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    print("Creating vector store...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=PERSIST_DIR
    )


    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    llm = Ollama(model=LLM_MODEL)
    
    prompt_template = """
    Answer the question based ONLY on the following context:

    {context}

    Question: {question}
    """
    
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("RAG chain created. You can now ask questions.")
    return rag_chain

def main():
    """
    Main function to run the Q&A loop.
    """
    try:
        rag_chain = build_rag_chain()
        
        while True:
            query = input("\nAsk a question (or 'exit' to quit): ")
            
            if query.lower() == 'exit':
                print("Exiting...")
                break
            
            if not query.strip():
                print("Please enter a question.")
                continue

            answer = rag_chain.invoke(query)
            print("\nAnswer:", answer)

    except FileNotFoundError:
        print(f"Error: The file '{DATA_FILE}' was not found.")
        print("Please make sure 'speech.txt' is in the same directory.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Please ensure Ollama is running ('ollama serve') and you have pulled the 'mistral' model ('ollama pull mistral').")
        sys.exit(1)

if __name__ == "__main__":
    main()