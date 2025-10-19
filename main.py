import os
import argparse
import sys
from rag_cli.env_check import check_environment
from rag_cli.setup_rag import setup_rag_system, setup_rag_system_local
from rag_cli.query import interactive_mode, query_rag


def main():
    if not check_environment():
        sys.exit(1)

    print("\n===============================")
    print("🔹 RAG CLI - Setup Menu")
    print("===============================\n")
    print("[1] Load documents from URL(s)")
    print("[2] Load local markdown folder")
    print("[3] Check environment")
    print("[4] Quit\n")

    choice = input("Select an option: ").strip()

    if choice == "1":
        urls = [u.strip() for u in input("Enter URLs (comma-separated): ").split(",")]
        graph = setup_rag_system(urls)

    elif choice == "2":
        print("\n🔍 Scanning for local markdown folders...\n")
        candidates = []
        for name in ["docs", "content", "notes", "markdown", "articles"]:
            if os.path.isdir(name):
                candidates.append(name)

        if candidates:
            print("Found the following possible folders:")
            for i, c in enumerate(candidates, 1):
                print(f" [{i}] {c}")
            print(f" [{len(candidates)+1}] Enter a custom path")

            selection = input("\nSelect a folder number or enter a custom path: ").strip()

            try:
                idx = int(selection)
                if 1 <= idx <= len(candidates):
                    folder = candidates[idx - 1]
                else:
                    folder = input("\nEnter your custom folder path: ").strip()
            except ValueError:
                folder = selection  # manual entry
        else:
            folder = input("\nNo common markdown folders found. Enter a folder path manually: ").strip()

        graph = setup_rag_system_local(folder)

    elif choice == "3":
        check_environment()
        return

    else:
        print("Goodbye!")
        return

    if not graph:
        print("❌ Failed to set up RAG system.")
        sys.exit(1)

    print("\n✅ Setup complete. Entering interactive mode...\n")
    interactive_mode(graph)


if __name__ == "__main__":
    main()
