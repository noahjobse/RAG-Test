def query_rag(graph, question, verbose=False):
    try:
        print(f"\nQuestion: {question}")
        result = graph.invoke({"question": question})
        print(f"\nAnswer:\n{result['answer']}")
    except Exception as e:
        print(f"Error querying RAG system: {e}")


def interactive_mode(graph):
    print("\nInteractive Mode - Ask questions below (type 'exit' to quit)\n")

    while True:
        try:
            question = input(">>> ").strip()
            if question.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
            elif question:
                query_rag(graph, question)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
