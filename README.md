# RAG-Powered Multi-Agent Q&A System

## Overview
This project implements a knowledge assistant that answers questions based on a collection of documents using Retrieval-Augmented Generation (RAG). It features an agentic workflow that routes specific queries to dedicated tools for calculations and definitions, enhancing its capabilities beyond standard Q&A. The system uses Pinecone as a vector database, Groq API as the language model, and Streamlit for the user interface.

## Features

- **Data Ingestion**: Ingests documents and stores their vector embeddings in Pinecone for efficient retrieval.
- **Vector Store & Retrieval**: Utilizes Pinecone to perform similarity searches and retrieve the most relevant document chunks for a given query.
- **LLM Integration**: Employs the Groq API (LLaMA3-8b-8192 model) to generate natural-language answers based on the retrieved context.
- **Agentic Workflow**: Routes queries containing keywords like "calculate" or "define" to specific tools (CalculatorTool and DefineTool), otherwise uses the RAG pipeline.
- **Demo Interface**: Provides a user-friendly web interface built with Streamlit to interact with the system.

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- Required libraries: langchain, pinecone-client, groq, streamlit, python-dotenv
- API keys for Pinecone and Groq

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/atharvmohanjadhav/RAG-powered-Multi-Agent-Q-A-Assistant.git
   cd RAG-powered-Multi-Agent-Q-A-Assistant
    ```


2. Install the required dependencies:

     
        pip install -r requirements.txt


3. Configure environment variables:

        Create a .env file in the project root.
        Add your Pinecone and Groq API keys:
        PINECONE_API_KEY=your-pinecone-api-key
        GROQ_API_KEY=your-groq-api-key


## Usage

1. Start the Streamlit application:
        streamlit run app.py


2. Open your web browser and navigate to the provided local URL (usually ``http://localhost:8501``).

3. Enter your question in the input field and submit.

    The interface will display:

    ``The tool or agent branch used (e.g., "Dictionary Tool", "Calculator Tool", or "RAG Pipeline").``

    ``The retrieved context snippets (if applicable).``

    ``The final answer.``


    Example Queries

    - "What is AI?"
    - "Calculate 15 % 200"
    - "Define photosynthesis"

## Architecture
### The system is structured as follows:

- Data Ingestion: Documents are processed and their embeddings are stored in Pinecone via ingest.py.
- Query Handling: The handle_query function in agents.py determines the appropriate action:

    If the query contains "define", it uses the DefineTool from tools.py.
    If the query contains "calculate", it uses the CalculatorTool from tools.py.
    Otherwise, it retrieves relevant document chunks from Pinecone using the retriever and generates an answer with the Groq API.


- User Interface: ``Streamlit in app.py`` provides an interactive web interface for users to input queries and view results.

# Key Design Choices

- **Pinecone:** Chosen for its scalability and efficiency in handling vector similarity searches, making it ideal for storing and retrieving document embeddings.
- **Groq API:** Selected for its high-performance LLaMA3-8b-8192 model, providing accurate and contextually relevant answer generation.
- **Streamlit:** Used for its simplicity and rapid development capabilities, enabling a clean and interactive UI with minimal effort.
- **Modular Tools:** Separate DefineTool and CalculatorTool functions in tools.py allow for extensibility and clear separation of concerns.

# Project Structure

1. **agents.py:** Contains the handle_query function for routing queries and managing the RAG pipeline.
2. **tools.py:** Defines the DefineTool and CalculatorTool for handling definition and calculation queries.
3. **ingest.py:** Handles document ingestion and sets up the Pinecone retriever.
4. **app.py:** Implements the Streamlit web interface.

# How to Run

Ensure all prerequisites are installed and API keys are configured in the .env file.

Run the Streamlit app:

        streamlit run app.py


Interact with the application through the web interface at ``http://localhost:8501``.

