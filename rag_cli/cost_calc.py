"""
rag_cli/cost_calc.py
Estimate OpenAI API costs for embeddings and chat models.
"""

import tiktoken
from typing import Literal

# === OFFICIAL 2025 PRICING (per 1M tokens) ===
PRICES = {
    "embeddings": {
        "text-embedding-3-small": 0.02 / 1_000_000,   # $ per token
        "text-embedding-3-large": 0.13 / 1_000_000,
    },
    "chat": {
        "gpt-5-mini": {"input": 0.25 / 1_000_000, "output": 2.00 / 1_000_000},
        "gpt-5": {"input": 1.25 / 1_000_000, "output": 10.00 / 1_000_000},
    }
}


def count_tokens(text: str, model="text-embedding-3-small") -> int:
    """Return number of tokens for text using tiktoken encoding."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def estimate_embedding_cost(text: str, model="text-embedding-3-small") -> dict:
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


def estimate_chat_cost(prompt_tokens: int, response_tokens: int,
                       model: Literal["gpt-5-mini", "gpt-5"] = "gpt-5-mini") -> dict:
    """Estimate chat cost for given input/output token counts."""
    p = PRICES["chat"][model]
    total = (prompt_tokens * p["input"]) + (response_tokens * p["output"])
    return {
        "model": model,
        "input_tokens": prompt_tokens,
        "output_tokens": response_tokens,
        "cost_usd": round(total, 6)
    }


def estimate_folder_embedding_cost(docs, model="text-embedding-3-small"):
    """Estimate cost for embedding all Documents in a folder."""
    total_tokens = sum(count_tokens(doc.page_content, model) for doc in docs)
    total_cost = total_tokens * PRICES["embeddings"][model]
    return {
        "model": model,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(total_cost, 6)
    }
