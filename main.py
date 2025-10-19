import os
import sys
from datetime import datetime
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from rag_cli.env_check import check_environment
from rag_cli.setup_rag import setup_rag_system, setup_rag_system_local
from rag_cli.query import interactive_mode
from rag_cli.cost_calc import estimate_folder_embedding_cost


# ============================================================
# Folder Navigator
# ============================================================
def select_local_folder(start_path=".") -> str:
    """
    Interactive folder selector that lets you navigate inside subdirectories.
    Use numbers to open a folder, '..' to go up, or 'select' to choose the current folder.
    """
    current = os.path.abspath(start_path)

    while True:
        print(f"\n📁 Current folder: {current}\n")

        subdirs = [d for d in os.listdir(current) if os.path.isdir(os.path.join(current, d))]
        if not subdirs:
            print("(No subfolders found here)")
        else:
            for i, d in enumerate(subdirs, 1):
                print(f" [{i}] {d}")

        print("\nCommands:")
        print("  Enter a number to open a folder")
        print("  '..' to go up a level")
        print("  'select' to use this folder")
        print("  'exit' to cancel\n")

        choice = input("Your choice: ").strip()

        if choice == "exit":
            return None
        elif choice == "select":
            return current
        elif choice == "..":
            parent = os.path.dirname(current)
            if parent == current:
                print("🚫 Already at the root directory.")
            else:
                current = parent
        else:
            try:
                idx = int(choice)
                if 1 <= idx <= len(subdirs):
                    current = os.path.join(current, subdirs[idx - 1])
                else:
                    print("❌ Invalid number.")
            except ValueError:
                possible = os.path.join(current, choice)
                if os.path.isdir(possible):
                    current = possible
                elif os.path.isdir(choice):
                    current = os.path.abspath(choice)
                else:
                    print("❌ Invalid input.")


# ============================================================
# Main CLI
# ============================================================
def main():
    if not check_environment():
        sys.exit(1)

    print("\n===============================")
    print("🔹 RAG CLI - Setup Menu")
    print("===============================\n")
    print("[1] Load documents from URL(s)")
    print("[2] Load local markdown folder")
    print("[3] Check environment")
    print("[4] Quit")
    print("[5] 💰 Estimate embedding cost for local folder\n")

    choice = input("Select an option: ").strip()

    # --- Option 1: URLs -------------------------------------------------------
    if choice == "1":
        urls = [u.strip() for u in input("Enter URLs (comma-separated): ").split(",")]
        graph = setup_rag_system(urls)

    # --- Option 2: Local markdown folder -------------------------------------
    elif choice == "2":
        folder = select_local_folder()
        graph = setup_rag_system_local(folder)

    # --- Option 3: Check environment -----------------------------------------
    elif choice == "3":
        check_environment()
        return

    # --- Option 5: Cost estimation -------------------------------------------
    elif choice == "5":
        folder = select_local_folder()
        if not folder:
            print("❌ No folder selected.")
            return

        if not os.path.isdir(folder):
            print("❌ That path does not exist or is not a directory.")
            return

        # Setup log file
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "embedding_cost.log")

        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n=== RAG CLI Cost Estimation — {datetime.now().isoformat()} ===\n")
            log.write(f"Selected folder: {folder}\n")

        print(f"\n🔍 Loading markdown files from {folder}...\n")

        loaded_docs = []
        skipped_files = []

        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(".md"):
                    path = os.path.join(root, f)
                    try:
                        with open(path, "r", encoding="utf-8") as fp:
                            text = fp.read()
                            loaded_docs.append(Document(page_content=text, metadata={"source": path}))
                    except Exception as e:
                        skipped_files.append((path, str(e)))

        if skipped_files:
            print(f"⚠️ Skipped {len(skipped_files)} files due to read errors.")
            for p, err in skipped_files[:3]:
                print(f"   - {os.path.basename(p)} → {err.splitlines()[0]}")
            if len(skipped_files) > 3:
                print(f"   ...and {len(skipped_files) - 3} more.\n")

        if not loaded_docs:
            print("⚠️ No readable markdown files found.")
            return

        print(f"✅ Loaded {len(loaded_docs)} valid markdown files.\n")

        try:
            print("💰 Calculating estimated embedding cost...\n")
            results = []

            for model in ["text-embedding-3-small", "text-embedding-3-large"]:
                result = estimate_folder_embedding_cost(loaded_docs, model=model)
                results.append(result)
                print(f"  • {model}:")
                print(f"     Tokens: {result['total_tokens']:,}")
                print(f"     Cost:   ${result['estimated_cost_usd']} USD\n")

            # Write results to log
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(f"Processed {len(loaded_docs)} documents.\n")
                if skipped_files:
                    log.write(f"Skipped {len(skipped_files)} files due to read errors.\n")
                for result in results:
                    log.write(f"Model: {result['model']}\n")
                    log.write(f"  Tokens: {result['total_tokens']}\n")
                    log.write(f"  Estimated cost: ${result['estimated_cost_usd']}\n")
            print(f"🪵 Log written to {log_path}")

        except Exception as e:
            err_msg = f"❌ Error estimating cost: {e}"
            print(err_msg)
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(err_msg + "\n")
        return

    # --- Option 4 or invalid --------------------------------------------------
    else:
        print("Goodbye!")
        return

    # --- Proceed to interactive query mode -----------------------------------
    if not graph:
        print("❌ Failed to set up RAG system.")
        sys.exit(1)

    print("\n✅ Setup complete. Entering interactive mode...\n")
    interactive_mode(graph)


if __name__ == "__main__":
    main()
