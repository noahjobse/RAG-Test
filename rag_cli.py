#!/usr/bin/env python3
"""
RAG CLI Tool - A command-line interface for Retrieval-Augmented Generation

This tool allows you to query documents (from web URLs or local folders)
using Retrieval-Augmented Generation (RAG) with customizable options
for models, chunk sizes, and overlap parameters.
"""

import argparse
import os
import sys
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# ============================================================
# ENVIRONMENT CHECK
# ============================================================
def check_environment():
    """Check if required environment variables are set."""
    if not os.getenv("USER_AGENT"):
        print("❌ Missing USER_AGENT in environment")
        print("   Set it in your .env file or environment variables")
        return False
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Missing OPENAI_API_KEY in environment")
        print("   Set it in your .env file or environment variables")
        return False
    
    return True


# ============================================================
# DOCUMENT SUMMARIZER
# ============================================================
def summarize_documents(docs, all_splits):
    """Generate a summary of the loaded documents using GPT."""
    try:
        from langchain.chat_models import init_chat_model
        
        llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        
        doc_summary = f"""
Document Summary:
- Number of documents loaded: {len(docs)}
- Total text chunks created: {len(all_splits)}
- Average chunk size: {sum(len(chunk.page_content) for chunk in all_splits) // len(all_splits)} characters

Document sources:
"""
        for i, doc in enumerate(docs, 1):
            doc_summary += f"- Document {i}: {doc.metadata.get('source', 'Unknown source')}\n"
        
        sample_content = "\n\n".join([chunk.page_content[:500] for chunk in all_splits[:3]])
        
        summary_prompt = f"""
Based on the following document content, provide a brief 2–3 sentence summary:

{doc_summary}

Sample content from the documents:
{sample_content}
"""
        response = llm.invoke(summary_prompt)
        return response.content
        
    except Exception as e:
        return f"Error generating summary: {e}"


# ============================================================
# RAG SETUP — WEB MODE
# ============================================================
def setup_rag_system(urls: List[str], chunk_size: int = 1000, chunk_overlap: int = 200):
    """Set up RAG system using web URLs."""
    try:
        from langchain.chat_models import init_chat_model
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.document_loaders import WebBaseLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.vectorstores import InMemoryVectorStore
        from langchain import hub
        from langgraph.graph import START, StateGraph
        from langchain_core.documents import Document
        from typing_extensions import TypedDict
        import bs4

        print(f"Setting up RAG system from URLs...")
        print(f"   URLs: {urls}")
        print(f"   Chunk size: {chunk_size}, overlap: {chunk_overlap}")

        llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        print("Loading documents...")
        loader = WebBaseLoader(
            web_paths=urls,
            bs_kwargs={"parse_only": bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))}
        )
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_splits = text_splitter.split_documents(docs)

        print("Generating summary...")
        print(summarize_documents(docs, all_splits))

        print("Embedding chunks...")
        vector_store = InMemoryVectorStore(embeddings)
        _ = vector_store.add_documents(all_splits)

        prompt = hub.pull("rlm/rag-prompt")

        class State(TypedDict):
            question: str
            context: List[Document]
            answer: str

        def retrieve(state: State):
            docs = vector_store.similarity_search(state["question"])
            return {"context": docs}

        def generate(state: State):
            context_text = "\n\n".join(doc.page_content for doc in state["context"])
            messages = prompt.invoke({"question": state["question"], "context": context_text})
            response = llm.invoke(messages)
            return {"answer": response.content}

        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()

        print("✅ RAG system ready from URLs.")
        return graph

    except Exception as e:
        print(f"Error setting up RAG system: {e}")
        return None


# ============================================================
# RAG SETUP — LOCAL MARKDOWN MODE
# ============================================================
def setup_rag_system_local(folder_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Set up RAG system using local Markdown (.md) files."""
    try:
        from langchain.chat_models import init_chat_model
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.document_loaders import DirectoryLoader, TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.vectorstores import InMemoryVectorStore
        from langchain import hub
        from langgraph.graph import START, StateGraph
        from langchain_core.documents import Document
        from typing_extensions import TypedDict

        print(f"Setting up RAG system from local folder: {folder_path}")
        print(f"   Chunk size: {chunk_size}, overlap: {chunk_overlap}")

        loader = DirectoryLoader(folder_path, glob="**/*.md", loader_cls=TextLoader)
        docs = loader.load()
        print(f"   Loaded {len(docs)} markdown documents.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_splits = text_splitter.split_documents(docs)
        print(f"   Split into {len(all_splits)} chunks.")

        print("Generating summary...")
        print(summarize_documents(docs, all_splits))

        llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        vector_store = InMemoryVectorStore(embeddings)
        _ = vector_store.add_documents(all_splits)

        prompt = hub.pull("rlm/rag-prompt")

        class State(TypedDict):
            question: str
            context: List[Document]
            answer: str

        def retrieve(state: State):
            docs = vector_store.similarity_search(state["question"])
            return {"context": docs}

        def generate(state: State):
            context_text = "\n\n".join(doc.page_content for doc in state["context"])
            messages = prompt.invoke({"question": state["question"], "context": context_text})
            response = llm.invoke(messages)
            return {"answer": response.content}

        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()

        print("✅ RAG system ready from local markdown folder.")
        return graph

    except Exception as e:
        print(f"Error setting up local RAG system: {e}")
        return None


# ============================================================
# QUERY FUNCTION
# ============================================================
def query_rag(graph, question: str, verbose: bool = False):
    """Query the RAG system with a question."""
    try:
        print(f"\nQuestion: {question}")
        result = graph.invoke({"question": question})
        
        if verbose:
            print(f"\nRetrieved {len(result.get('context', []))} context documents")

        print(f"\nAnswer:\n{result['answer']}")
        return result['answer']
    except Exception as e:
        print(f"Error querying RAG system: {e}")
        return None


# ============================================================
# INTERACTIVE MODE
# ============================================================
def interactive_mode(graph):
    """Run the CLI in interactive mode."""
    print("\nInteractive RAG Query Mode")
    print("   Type 'quit' or 'exit' to stop")
    print("   Type 'help' for available commands\n")

    while True:
        try:
            question = input("\nEnter your question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif question.lower() == 'help':
                print("Commands: help | quit | summary | any question")
                continue
            elif not question:
                continue
            query_rag(graph, question)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


# ============================================================
# MAIN CLI ENTRY
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="RAG CLI Tool - Query documents using Retrieval-Augmented Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query a single question
  python rag_cli.py --question "What is Task Decomposition?"

  # Interactive mode with custom URLs
  python rag_cli.py --interactive --urls "https://example.com/article1"

  # Local markdown folder
  python rag_cli.py --folder ./docs --interactive
        """
    )

    parser.add_argument("--question", "-q", type=str, help="Single question to ask the RAG system")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--urls", "-u", nargs="+", help="List of URLs to load documents from")
    parser.add_argument("--folder", "-f", type=str, help="Path to folder containing markdown docs")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Text chunk size (default: 1000)")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Text chunk overlap (default: 200)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--check-env", action="store_true", help="Check environment variables and exit")

    args = parser.parse_args()

    if args.check_env:
        if check_environment():
            print("✅ Environment variables are properly configured")
        else:
            sys.exit(1)
        return

    if not check_environment():
        sys.exit(1)

    # Choose between URL or folder mode
    if args.folder:
        graph = setup_rag_system_local(args.folder, args.chunk_size, args.chunk_overlap)
    else:
        urls = args.urls or ["https://lilianweng.github.io/posts/2023-06-23-agent/"]
        graph = setup_rag_system(urls, args.chunk_size, args.chunk_overlap)

    if not graph:
        sys.exit(1)

    # Run modes
    if args.question:
        query_rag(graph, args.question, args.verbose)
    else:
        interactive_mode(graph)


if __name__ == "__main__":
    main()
