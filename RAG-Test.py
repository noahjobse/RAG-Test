# rag_test.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get user agent from .env file
if not os.getenv("USER_AGENT"):
    raise EnvironmentError("Missing USER_AGENT in environment")

# Get API keys from .env file
if not os.environ.get("OPENAI_API_KEY"):
    raise EnvironmentError("Missing OPENAI_API_KEY in environment")

# Imports
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
import bs4

# Initialize LLM and Embeddings
llm = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Load content
loader = WebBaseLoader(
    web_paths=["https://lilianweng.github.io/posts/2023-06-23-agent/"],
    bs_kwargs={"parse_only": bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))}
)
docs = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Embed and store in vector store
vector_store = InMemoryVectorStore(embeddings)
_ = vector_store.add_documents(all_splits)

# Define LangGraph RAG pipeline
prompt = hub.pull("rlm/rag-prompt")

# Define the state type - question in, context retreived, answer out
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Retrieves context based on the question
def retrieve(state: State):
    docs = vector_store.similarity_search(state["question"])
    return {"context": docs}

# Constructs an answer based on the retrieved context and the question
def generate(state: State):
    context_text = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": context_text})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Build the state graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Run test query
if __name__ == "__main__":
    question = "What is Task Decomposition?"
    result = graph.invoke({"question": question})
    print("\nQuestion:", question)
    print("\nAnswer:\n", result["answer"])
