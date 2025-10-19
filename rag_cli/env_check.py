import os
from dotenv import load_dotenv

load_dotenv()

def check_environment():
    """Check if required environment variables are set."""
    missing = []

    if not os.getenv("USER_AGENT"):
        missing.append("USER_AGENT")
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")

    if missing:
        print(f"❌ Missing required environment variables: {', '.join(missing)}")
        print("   Add them to your .env file or environment variables.")
        return False

    print("✅ Environment variables are properly configured")
    return True
