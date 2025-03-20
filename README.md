# Lamb Knowledge Base Server (lamb-kb-server)

A dedicated knowledge base server designed to provide robust vector database functionality for the LAMB project and to serve as a Model Context Protocol (MCP) server. It uses ChromaDB for vector database storage and FastAPI to create an API that allows the LAMB project to access knowledge databases.

## Setup and Installation

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd lamb-kb-server
```

2. Install the required dependencies:
```bash
cd backend
pip install -r requirements.txt
```

3. Environment variables:
   - Copy `.env.example` to `.env`
   - Modify the API key as needed (default: "0p3n-w3bu!")
   - Configure embedding model settings (see Embeddings Configuration section)

### Running the Server

```bash
cd backend
python start.py
```

The server will run on http://localhost:9090 by default.

## API Authentication

All API calls require a Bearer token for authentication. The token must match the `LAMB_API_KEY` environment variable.

Example request:
```bash
curl -H 'Authorization: Bearer 0p3n-w3bu!' http://localhost:9090/
```

## Features

### Core Functionality

- **Collections Management**: Create, view, update and manage document collections
- **Document Ingestion**: Process and store documents with vectorized content
- **Similarity Search**: Query collections to find semantically similar content
- **Static File Serving**: Serve original documents via URL references

### Plugin System

The server implements a flexible plugin architecture for both ingestion and querying:

#### Ingestion Plugins

Plugins for processing different document types with configurable chunking strategies:

- **simple_ingest**: Processes text files with options for character, word, or line-based chunking
- Support for custom chunking parameters:
  - `chunk_size`: Size of each chunk
  - `chunk_unit`: Unit for chunking (`char`, `word`, or `line`)
  - `chunk_overlap`: Overlap between chunks

#### Query Plugins

Plugins for different query strategies:

- **simple_query**: Performs similarity searches with configurable parameters:
  - `top_k`: Number of results to return
  - `threshold`: Minimum similarity threshold

### Embeddings Configuration

The system supports multiple embedding providers:

1. **Local Embeddings** (default)
   - Uses sentence-transformers models locally
   - Example configuration in `.env`:
     ```
     EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2
     EMBEDDINGS_VENDOR=local
     EMBEDDINGS_APIKEY=
     ```

2. **OpenAI Embeddings**
   - Uses OpenAI's embedding API
   - Requires an API key
   - Example configuration in `.env`:
     ```
     EMBEDDINGS_MODEL=text-embedding-3-small
     EMBEDDINGS_VENDOR=openai
     EMBEDDINGS_APIKEY=your-openai-key-here
     ```

When creating collections, you can specify the embedding configuration or use "default" to inherit from environment variables:

```json
"embeddings_model": {
  "model": "default",
  "vendor": "default",
  "apikey": "default"
}
```

## API Examples

### Creating a Collection

```bash
curl -X POST 'http://localhost:9090/collections' \
  -H 'Authorization: Bearer 0p3n-w3bu!' \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "my-knowledge-base",
    "description": "My first knowledge base",
    "owner": "user1",
    "visibility": "private",
    "embeddings_model": {
      "model": "default",
      "vendor": "default",
      "apikey": "default"
    }
  }'
```

### Ingesting a File

```bash
curl -X POST 'http://localhost:9090/collections/1/ingest_file' \
  -H 'Authorization: Bearer 0p3n-w3bu!' \
  -F 'file=@/path/to/document.txt' \
  -F 'plugin_name=simple_ingest' \
  -F 'plugin_params={"chunk_size":1000,"chunk_unit":"char","chunk_overlap":200}'
```

### Querying a Collection

```bash
curl -X POST 'http://localhost:9090/collections/1/query' \
  -H 'Authorization: Bearer 0p3n-w3bu!' \
  -H 'Content-Type: application/json' \
  -d '{
    "query_text": "What is machine learning?",
    "top_k": 5,
    "threshold": 0.5,
    "plugin_params": {}
  }'
```

## Testing

The repository includes test scripts to verify functionality:

- **test.py**: A comprehensive test script that demonstrates creating collections, ingesting documents with different chunking strategies, and performing queries
- **params.json**: Configuration for the test script

To run the tests:

```bash
cd backend
python test.py
```
