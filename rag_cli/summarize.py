def summarize_documents(docs, all_splits):
    """Generate a short summary of the loaded documents using GPT."""
    try:
        from langchain.chat_models import init_chat_model

        llm = init_chat_model("gpt-5-mini", model_provider="openai")

        summary_prompt = f"""
        You are summarizing a document collection.

        - Documents loaded: {len(docs)}
        - Total chunks: {len(all_splits)}
        - Average chunk size: {sum(len(c.page_content) for c in all_splits) // len(all_splits)} characters

        Write a concise summary (2–3 sentences) describing their overall content.
        """

        response = llm.invoke(summary_prompt)
        return response.content

    except Exception as e:
        return f"Error generating summary: {e}"
