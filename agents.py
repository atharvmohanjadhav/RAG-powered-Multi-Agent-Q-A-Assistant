
import os
from dotenv import load_dotenv
from tools import DefineTool, CalculatorTool
from ingest import retriever  # Pinecone retriever from your ingest.py
from langchain_groq import ChatGroq  # or use OpenAI if preferred

load_dotenv()

chat = ChatGroq(
    temperature=0,
    model_name="llama3-8b-8192",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

def handle_query(user_query: str):
    """
    Routes query to the appropriate tool or uses RAG pipeline.
    Returns a dict with the selected tool, retrieved context (if any), and final answer.
    """
    log = {
        "tool": None,
        "context": [],
        "answer": None
    }

    query = user_query.strip().lower()

    # Tool Routing
    if "define" in query:
        log["tool"] = "Dictionary Tool"
        define_tool = DefineTool()
        term = user_query.split(" ", 1)[1]
        log["answer"] = define_tool.run(term)
        print("Tool selected: Dictionary Tool")

    elif "calculate" in query:
        log["tool"] = "Calculator Tool"
        calc_tool = CalculatorTool()
        expression = user_query.split(" ", 1)[1]
        log["answer"] = calc_tool.run(expression)
        print("Tool selected: Calculator Tool")

    else:
        # RAG Flow: Retrieval + Generation
        log["tool"] = "RAG Pipeline"
        docs = retriever.invoke(user_query)
        context_chunks = [doc.page_content if hasattr(doc, "page_content") else doc for doc in docs]

        log["context"] = context_chunks
        combined_context = "\n\n".join(context_chunks)

        prompt = f"You are a helpful assistant. Based on the following context, answer the question.\n\nContext:\n{combined_context}\n\nQuestion: {user_query}"

        response = chat.invoke(prompt)
        log["answer"] = response.content
        print("Tool selected: RAG (LLM)")

    return log
