# Simple RAG Q&A System with LangChain & Ollama

This project is a simple, command-line Q&A system that answers questions based *only* on the content of a provided speech by Dr. B.R. Ambedkar.

It uses a RAG (Retrieval-Augmented Generation) pipeline built with LangChain. All components are 100% local, free, and require no API keys:
* **LLM:** Ollama (running Mistral 7B)
* **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
* **Vector Store:** ChromaDB

---

## Setup Instructions

You must have [Ollama](https://ollama.com/) installed on your system for this to work.

### Step 1: Install Ollama and Pull Mistral

1.  **Install Ollama:** Follow the instructions for your OS at [ollama.com](https://ollama.com/).
2.  **Verify Installation:** Open your terminal and run:
    ```bash
    ollama --version
    ```
3.  **Pull the Mistral Model:** This will download the 7B model (approx. 4.1 GB) to your local machine.
    ```bash
    ollama pull mistral
    ```

### Step 2: Set Up the Python Project

1.  **Clone the Repository:**
    ```bash
    git clone <https://github.com/manthan-jsharma/AmbedkarGPT-Intern-Task>
    cd <AmbedkarGPT-Intern-Task>
    ```

2.  **Create a Virtual Environment** (Recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows, use: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## How to Run

Once setup is complete, you can start the Q&A system from your terminal.

1.  **Run the script:**
    ```bash
    python main.py
    ```

2.  **Wait for Setup:** The first time you run it, the script will:
    * Download the `all-MiniLM-L6-v2` embedding model (a few hundred MB).
    * Load `speech.txt`, split it, create embeddings, and save them to a local `./chroma_db` directory.

3.  **Ask Questions:** Once you see the prompt, you can ask questions based on the text. The system will retrieve the relevant context and generate an answer.

    ```
    Loading document...
    Splitting text...
    Creating vector store...
    RAG chain created. You can now ask questions.
    
    Ask a question (or 'exit' to quit): What is the real remedy?
    
    Answer: The real remedy is to destroy the belief in the sanctity of the shastras.
    
    Ask a question (or 'exit' to quit): What is the work of social reform compared to?
    
    Answer: The work of social reform is compared to the work of a gardener who is constantly pruning the leaves and branches of a tree without ever attacking the roots.
    
    Ask a question (or 'exit' to quit): exit
    ```
