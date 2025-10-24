# RAG CLI Tool

A command-line interface for Retrieval-Augmented Generation (RAG) that enables querying documents using AI-powered search and answer generation.

## Overview

The RAG CLI Tool provides a robust solution for document-based question answering by combining the power of large language models with semantic search capabilities. It processes web documents, creates vector embeddings, and generates contextual answers based on retrieved information.

## Features

- **Document Retrieval**: Load and process documents from web URLs with configurable parsing
- **AI-Powered Answers**: Generate contextual answers using OpenAI's GPT models
- **Interactive Mode**: Chat-like interface for multiple queries with command history
- **Customizable Parameters**: Adjust chunk sizes, URLs, and other processing parameters
- **Verbose Output**: Detailed information about the retrieval and processing pipeline
- **Environment Management**: Simple configuration through environment variables
- **Error Handling**: Comprehensive error checking and user-friendly error messages

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Internet connection for document loading

### Setup Instructions

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   Create a `.env` file in the project directory with:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   USER_AGENT=your_user_agent_string_here
   ```

## Usage

### Basic Commands

**Environment verification**:
```bash
python rag_cli.py --check-env
```

**Single question query**:
```bash
python rag_cli.py --question "What is Task Decomposition?"
```

**Interactive mode** (default behavior):
```bash
python rag_cli.py --interactive
# or simply:
python rag_cli.py
```

### Advanced Configuration

**Custom document sources**:
```bash
python rag_cli.py --urls "https://example.com/article1" "https://example.com/article2" --interactive
```

**Optimized chunk settings**:
```bash
python rag_cli.py --chunk-size 500 --chunk-overlap 100 --question "What is AI?"
```

**Detailed processing information**:
```bash
python rag_cli.py --verbose --question "What is machine learning?"
```

### Interactive Mode Commands

When running in interactive mode, the following commands are available:
- Enter any question to receive an AI-generated answer
- `help` - Display available commands and usage information
- `quit`, `exit`, or `q` - Terminate the application

## Examples

### Example 1: Basic Query Processing
```bash
$ python rag_cli.py --question "What is Task Decomposition?"
Setting up RAG system...
   URLs: ['https://lilianweng.github.io/posts/2023-06-23-agent/']
   Chunk size: 1000
   Chunk overlap: 200
Loading documents from URLs...
   Loaded 1 documents
Splitting documents into chunks...
   Split into 15 chunks
Creating vector store...
RAG system ready!

Question: What is Task Decomposition?
Searching for relevant context...

Answer:
Task decomposition is a technique used in AI and machine learning where complex tasks are broken down into smaller, more manageable subtasks...
```

### Example 2: Interactive Session
```bash
$ python rag_cli.py --interactive
Setting up RAG system...
   URLs: ['https://lilianweng.github.io/posts/2023-06-23-agent/']
   Chunk size: 1000
   Chunk overlap: 200
Loading documents from URLs...
   Loaded 1 documents
Splitting documents into chunks...
   Split into 15 chunks
Creating vector store...
RAG system ready!

Interactive RAG Query Mode
   Type 'quit' or 'exit' to stop
   Type 'help' for available commands

Enter your question: What is an AI agent?
Searching for relevant context...

Answer:
An AI agent is an autonomous system that can perceive its environment, make decisions, and take actions to achieve specific goals...

Enter your question: quit
Goodbye!
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key for model access | Yes |
| `USER_AGENT` | User agent string for web scraping | Yes |

### Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--question` | `-q` | Single question to process | None |
| `--interactive` | `-i` | Enable interactive mode | False |
| `--urls` | `-u` | Document URLs to process | Lilian Weng's agent post |
| `--chunk-size` | | Text chunk size in characters | 1000 |
| `--chunk-overlap` | | Overlap between chunks | 200 |
| `--verbose` | `-v` | Enable detailed output | False |
| `--check-env` | | Verify environment configuration | False |

## Technical Architecture

### Processing Pipeline

1. **Document Loading**: Web scraping with BeautifulSoup for content extraction
2. **Text Processing**: Recursive character-based text splitting with configurable parameters
3. **Vector Embedding**: OpenAI's text-embedding-3-large model for semantic representation
4. **Similarity Search**: In-memory vector store for efficient document retrieval
5. **Answer Generation**: GPT-5-mini model with retrieved context for response generation

### Technology Stack

- **Language Model**: GPT-5-mini (OpenAI)
- **Embedding Model**: text-embedding-3-large (OpenAI)
- **Vector Store**: InMemoryVectorStore (LangChain)
- **Orchestration**: LangGraph for workflow management
- **Text Processing**: RecursiveCharacterTextSplitter
- **Web Scraping**: BeautifulSoup with custom selectors

## Performance Considerations

- **Memory Usage**: In-memory vector store requires sufficient RAM for large document collections
- **API Costs**: OpenAI API usage incurs costs based on token consumption
- **Processing Time**: Initial setup includes document loading and embedding generation
- **Network Dependency**: Requires internet access for document retrieval and API calls

## Troubleshooting

### Common Issues

1. **Missing API Key**: Ensure `OPENAI_API_KEY` is properly set in your environment
2. **Missing User Agent**: Configure `USER_AGENT` for web scraping compliance
3. **Import Errors**: Install all dependencies using the provided requirements.txt
4. **Network Connectivity**: Verify internet access for document loading and API calls
5. **Memory Limitations**: Consider reducing chunk size for large document collections

### Error Resolution

**Environment verification**:
```bash
python rag_cli.py --check-env
```

**Dependency installation**:
```bash
pip install -r requirements.txt
```

**Help documentation**:
```bash
python rag_cli.py --help
```

## Development

### Project Structure

```
RAG-Test/
├── rag_cli.py          # Main CLI application
├── requirements.txt    # Python dependencies
├── setup.py           # Package configuration
├── README.md          # Documentation
├── rag-cli.bat        # Windows execution script
├── RAG-Test.py        # Original implementation
└── RAG-Test.ipynb     # Jupyter notebook version
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add appropriate tests
5. Submit a pull request

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Support

For issues, questions, or contributions, please refer to the project documentation or create an issue in the repository.
