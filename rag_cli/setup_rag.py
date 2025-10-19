from typing import List
from rag_cli.summarize import summarize_documents

def setup_rag_system(urls: List[str], chunk_size=1000, chunk_overlap=200):
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

        print(f"Setting up RAG system from URLs: {urls}")
        llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        loader = WebBaseLoader(
            web_paths=urls,
            bs_kwargs={"parse_only": bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))}
        )
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_splits = text_splitter.split_documents(docs)

        print("Generating summary...")
        print(summarize_documents(docs, all_splits))

        vector_store = InMemoryVectorStore(embeddings)
        _ = vector_store.add_documents(all_splits)

        prompt = hub.pull("rlm/rag-prompt")

        class State(TypedDict):
            question: str
            context: list[Document]
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


def setup_rag_system_local(folder_path: str, chunk_size=1000, chunk_overlap=200):
    """Set up RAG system from local markdown files."""
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

        print(f"Loading local docs from {folder_path}...")
        loader = DirectoryLoader(folder_path, glob="**/*.md", loader_cls=TextLoader)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_splits = text_splitter.split_documents(docs)

        print("Generating summary...")
        print(summarize_documents(docs, all_splits))

        llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_store = InMemoryVectorStore(embeddings)
        _ = vector_store.add_documents(all_splits)

        prompt = hub.pull("rlm/rag-prompt")

        class State(TypedDict):
            question: str
            context: list[Document]
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

        print("✅ RAG system ready from local folder.")
        return graph

    except Exception as e:
        print(f"Error setting up local RAG system: {e}")
        return None
