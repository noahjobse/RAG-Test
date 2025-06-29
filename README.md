# RAG Test - Custom Chatbot for New and Niche Topics

This project explores Retrieval-Augmented Generation (RAG) using LangChain, LangGraph, and OpenAI GPT models. The goal is to develop a chatbot that can answer questions about brand new or highly specific technologies that ChatGPT and other LLMs do not cover out of the box.

## Why This Project

Language models are powerful, but they are limited to their training data. This makes them unreliable for:

• New tools, APIs, or libraries  
• Experimental technologies  
• Internal documents or proprietary workflows

This project tests a solution using RAG. It injects relevant documents into the model’s context in real time, enabling the LLM to generate grounded and up-to-date responses.

## What This Demo Does

1. Loads a blog post from Lilian Weng on multi-agent systems  
2. Splits the text into manageable chunks  
3. Embeds the chunks using OpenAI's text-embedding-3-large  
4. Stores them in memory using an InMemoryVectorStore  
5. Accepts natural-language questions from the user  
6. Retrieves relevant chunks using semantic search  
7. Feeds them to GPT-40-mini to generate an answer  
8. Optionally traces the process using LangSmith for visibility

## Tech Stack

LangChain  
LangGraph  
OpenAI GPT-4o-mini
OpenAI Embeddings (text-embedding-3-large)  
Langchain InMemoryVectorStore  
LangSmith (optional for debugging and trace inspection)

## Example Use Case

This method can be applied to any content not covered by public LLMs, such as:

• New research or whitepapers  
• Internal engineering documentation  
• Customer knowledge bases  
• Developer wikis or changelogs  
• Unreleased product specs

## Current Notebook

See RAG-Test.ipynb in the project directory for the full LangGraph pipeline.

## Future Ideas

• Replace in-memory store with persistent vector DB  
• Add UI with Streamlit or a web frontend  
• Support filtering, tags, or hybrid search  
• Allow real-time updates from new documents  
• Add authentication and deployment infrastructure

## License

MIT
