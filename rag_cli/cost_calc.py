"""
rag_cli/cost_calc.py
Estimate OpenAI API and Qdrant storage costs for embeddings and chat models.
"""

import tiktoken
from typing import Literal
from langchain_core.documents import Document


# ============================================================
# PRICING TABLE (Official 2025 rates per 1M tokens)
# ============================================================
PRICES = {
    "embeddings": {
        "text-embedding-3-small": 0.02 / 1_000_000,   # USD per token
        "text-embedding-3-large": 0.13 / 1_000_000,
    },
    "chat": {
        "gpt-5-mini": {"input": 0.25 / 1_000_000, "output": 2.00 / 1_000_000},
        "gpt-5": {"input": 1.25 / 1_000_000, "output": 10.00 / 1_000_000},
    }
}


# ============================================================
# CORE TOKEN + COST FUNCTIONS
# ============================================================

def count_tokens(text: str, model: str = "text-embedding-3-small") -> int:
    """Count tokens using the appropriate tokenizer for a model."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def estimate_embedding_cost(text: str, model: str = "text-embedding-3-small") -> dict:
    """Estimate total cost for embedding a given string."""
    tokens = count_tokens(text, model)
    cost_per_token = PRICES["embeddings"].get(model)
    if not cost_per_token:
        raise ValueError(f"Unknown embedding model: {model}")
    total = tokens * cost_per_token
    return {
        "model": model,
        "tokens": tokens,
        "cost_usd": round(total, 6)
    }


def estimate_chat_cost(
    prompt_tokens: int,
    response_tokens: int,
    model: Literal["gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> dict:
    """Estimate chat API cost for a given number of input/output tokens."""
    p = PRICES["chat"][model]
    total = (prompt_tokens * p["input"]) + (response_tokens * p["output"])
    return {
        "model": model,
        "input_tokens": prompt_tokens,
        "output_tokens": response_tokens,
        "cost_usd": round(total, 6)
    }


def estimate_folder_embedding_cost(docs: list[Document], model: str = "text-embedding-3-small") -> dict:
    """Estimate total embedding cost for all documents in a folder."""
    total_tokens = sum(count_tokens(doc.page_content, model) for doc in docs)
    total_cost = total_tokens * PRICES["embeddings"][model]
    return {
        "model": model,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(total_cost, 6)
    }


# ============================================================
# QDRANT STORAGE & COST PREVIEW
# ============================================================

def estimate_qdrant_embedding_plan(docs: list[Document], model: str = "text-embedding-3-small") -> dict:
    """
    Estimate tokens, embedding cost, and vector storage size before uploading to Qdrant.

    Args:
        docs (list[Document]): Loaded markdown or text Documents.
        model (str): Embedding model ("text-embedding-3-small" or "text-embedding-3-large").

    Returns:
        dict: Combined summary of estimated tokens, cost, and Qdrant vector storage.
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    total_tokens = sum(len(enc.encode(doc.page_content)) for doc in docs)
    cost_per_token = PRICES["embeddings"].get(model, 0)
    estimated_cost = total_tokens * cost_per_token

    # Model dimensions and storage math
    dim = 1536 if "small" in model else 3072
    bytes_per_vector = dim * 4  # float32 (4 bytes per value)
    storage_bytes = len(docs) * bytes_per_vector

    summary = {
        "model": model,
        "documents": len(docs),
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(estimated_cost, 6),
        "dimension": dim,
        "storage_mb": round(storage_bytes / 1e6, 3),
    }

    # Nicely formatted output for CLI
    print("\n🧮 Qdrant Embedding Plan Summary")
    print("-----------------------------------")
    print(f"Model: {model}")
    print(f"Documents: {summary['documents']}")
    print(f"Estimated tokens: {summary['total_tokens']:,}")
    print(f"Estimated embedding cost: ${summary['estimated_cost_usd']:.4f} USD")
    print(f"Vector dimension: {dim}")
    print(f"Estimated Qdrant storage: {summary['storage_mb']:.2f} MB")
    print("-----------------------------------\n")

    return summary


# ============================================================
# MASTER FUNCTION — Combined Summary for Both Models
# ============================================================

def summarize_embedding_options(docs: list[Document]) -> dict:
    """
    Print a full summary comparing small vs large embeddings for both cost and storage.
    """
    print("\n📊 Embedding Options Comparison")
    print("===================================")
    summaries = {}
    for model in ["text-embedding-3-small", "text-embedding-3-large"]:
        summaries[model] = estimate_qdrant_embedding_plan(docs, model=model)
    return summaries
