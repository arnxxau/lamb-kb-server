import os
import json
from typing import Dict, Any, List, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load the environment variables from .env file
    load_dotenv()
    print(f"INFO: Environment variables loaded from .env file")
    print(f"INFO: EMBEDDINGS_VENDOR={os.getenv('EMBEDDINGS_VENDOR')}")
    print(f"INFO: EMBEDDINGS_MODEL={os.getenv('EMBEDDINGS_MODEL')}")
except ImportError:
    print("WARNING: python-dotenv not installed, environment variables must be set manually")

from fastapi import Depends, FastAPI, HTTPException, status, Query, File, Form, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

# Database imports
from database.connection import init_databases, get_db, get_chroma_client
from database.models import Collection, Visibility, FileRegistry, FileStatus
from database.service import CollectionService
from schemas.collection import (
    CollectionCreate, 
    CollectionUpdate, 
    CollectionResponse, 
    CollectionList,
    EmbeddingsModel
)

# Import ingestion modules
from plugins.base import discover_plugins
from services.ingestion import IngestionService
from services.query import QueryService
from services.collections import CollectionsService
from schemas.ingestion import (
    IngestionPluginInfo,
    IngestFileRequest,
    IngestFileResponse,
    AddDocumentsRequest,
    AddDocumentsResponse
)

# Import query modules
from schemas.query import (
    QueryRequest,
    QueryResponse,
    QueryPluginInfo
)

# Import file registry schemas
class FileRegistryResponse(BaseModel):
    """Model for file registry entry response"""
    id: int = Field(..., description="ID of the file registry entry")
    collection_id: int = Field(..., description="ID of the collection")
    original_filename: str = Field(..., description="Original filename")
    file_path: str = Field(..., description="Path to the file on the server")
    file_url: str = Field(..., description="URL to access the file")
    file_size: int = Field(..., description="Size of the file in bytes")
    content_type: Optional[str] = Field(None, description="MIME type of the file")
    plugin_name: str = Field(..., description="Name of the ingestion plugin used")
    plugin_params: Dict[str, Any] = Field(..., description="Parameters used for ingestion")
    status: str = Field(..., description="Status of the file")
    document_count: int = Field(..., description="Number of documents created from this file")
    created_at: str = Field(..., description="Timestamp when the file was added")
    updated_at: str = Field(..., description="Timestamp when the file record was last updated")
    owner: str = Field(..., description="Owner of the file")

# Get API key from environment variable or use default
API_KEY = os.getenv("LAMB_API_KEY", "0p3n-w3bu!")

# Get default embeddings model configuration from environment variables
# Default to using Ollama with nomic-embed-text model
# For OpenAI models, the environment variables should be set accordingly
DEFAULT_EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text")
DEFAULT_EMBEDDINGS_VENDOR = os.getenv("EMBEDDINGS_VENDOR", "ollama")  # 'ollama', 'local', or 'openai'
DEFAULT_EMBEDDINGS_APIKEY = os.getenv("EMBEDDINGS_APIKEY", "")
# Default endpoint for Ollama
DEFAULT_EMBEDDINGS_ENDPOINT = os.getenv("EMBEDDINGS_ENDPOINT", "http://localhost:11434/api/embeddings")

# Response models
class HealthResponse(BaseModel):
    """Model for health check responses"""
    status: str = Field(..., description="Status of the server", example="ok")
    version: str = Field(..., description="Server version", example="0.1.0")

class MessageResponse(BaseModel):
    """Model for basic message responses"""
    message: str = Field(..., description="Response message", example="Hello World from the Lamb Knowledge Base Server!")

class DatabaseStatusResponse(BaseModel):
    """Model for database status response"""
    sqlite_status: Dict[str, Any] = Field(..., description="Status of SQLite database")
    chromadb_status: Dict[str, Any] = Field(..., description="Status of ChromaDB database")
    collections_count: int = Field(..., description="Number of collections")

# Initialize FastAPI app with detailed documentation
app = FastAPI(
    title="Lamb Knowledge Base Server",
    description="""A dedicated knowledge base server designed to provide robust vector database functionality 
    for the LAMB project and to serve as a Model Context Protocol (MCP) server.
    
    ## Authentication
    
    All API endpoints are secured with Bearer token authentication. The token must match 
    the `LAMB_API_KEY` environment variable (default: `0p3n-w3bu!`).
    
    Example:
    ```
    curl -H 'Authorization: Bearer 0p3n-w3bu!' http://localhost:9090/
    ```
    
    ## Features
    
    - Knowledge base management for LAMB Learning Assistants
    - Vector database services using ChromaDB
    - API access for the LAMB project
    - Model Context Protocol (MCP) compatibility
    """,
    version="0.1.0",
    contact={
        "name": "LAMB Project Team",
    },
    license_info={
        "name": "GNU General Public License v3.0",
        "url": "https://www.gnu.org/licenses/gpl-3.0.en.html"
    },
)

# Security scheme with documentation
security = HTTPBearer(
    scheme_name="Bearer Authentication",
    description="Enter the API token as a Bearer token",
    auto_error=True,
)

# Authentication dependency with detailed docstring
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the provided Bearer token against the API key.
    
    Args:
        credentials: The HTTP Authorization credentials containing the Bearer token
        
    Returns:
        The validated token string
        
    Raises:
        HTTPException: If the token is invalid or missing
    """
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


# Initialize databases on startup
@app.on_event("startup")
async def startup_event():
    """Initialize databases and perform sanity checks on startup."""
    print("Initializing databases...")
    init_status = init_databases()
    
    if init_status["errors"]:
        for error in init_status["errors"]:
            print(f"ERROR: {error}")
    else:
        print("Databases initialized successfully.")
    
    # Discover ingestion plugins
    print("Discovering ingestion plugins...")
    discover_plugins("plugins")
    print(f"Found {len(IngestionService.list_plugins())} ingestion plugins")
    
    # Ensure static directory exists
    IngestionService._ensure_dirs()


# Configure static files
static_dir = IngestionService.STATIC_DIR
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Query endpoints
@app.get(
    "/query/plugins",
    response_model=List[QueryPluginInfo],
    summary="List available query plugins",
    description="""Get a list of available query plugins.
    
    This endpoint returns a list of all registered query plugins with their metadata.
    
    Example:
    ```bash
    curl -X GET 'http://localhost:9090/query/plugins' \
      -H 'Authorization: Bearer 0p3n-w3bu!'
    ```
    """,
    tags=["Query"],
    responses={
        200: {"description": "List of available query plugins"},
        401: {"description": "Unauthorized - Invalid or missing authentication token"}
    }
)
async def list_query_plugins(
    token: str = Depends(verify_token)
):
    """List available query plugins.
    
    Args:
        token: Authentication token
        
    Returns:
        List of query plugins
    """
    return QueryService.list_plugins()


@app.post(
    "/collections/{collection_id}/query",
    response_model=QueryResponse,
    summary="Query a collection",
    description="""Query a collection using a specified plugin.
    
    This endpoint performs a query on a collection using the specified query plugin.
    
    Example for simple_query plugin:
    ```bash
    curl -X POST 'http://localhost:9090/collections/1/query' \
      -H 'Authorization: Bearer 0p3n-w3bu!' \
      -H 'Content-Type: application/json' \
      -d '{
        "query_text": "What is the capital of France?",
        "top_k": 5,
        "threshold": 0.5,
        "plugin_params": {}
      }'
    ```
    
    Parameters for simple_query plugin:
    - query_text: The text to query for
    - top_k: Number of results to return (default: 5)
    - threshold: Minimum similarity threshold (0-1) (default: 0.0)
    """,
    tags=["Query"],
    responses={
        200: {"description": "Query results"},
        400: {"description": "Invalid query parameters"},
        401: {"description": "Unauthorized - Invalid or missing authentication token"},
        404: {"description": "Collection or query plugin not found"}
    }
)
async def query_collection(
    collection_id: int,
    request: QueryRequest,
    plugin_name: str = Query("simple_query", description="Name of the query plugin to use"),
    token: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Query a collection using a specified plugin.
    
    Args:
        collection_id: ID of the collection to query
        request: Query request parameters
        plugin_name: Name of the query plugin to use
        token: Authentication token
        db: Database session
        
    Returns:
        Query results
        
    Raises:
        HTTPException: If collection not found, plugin not found, or query fails
    """
    # Check if collection exists in SQLite
    collection = CollectionService.get_collection(db, collection_id)
    if not collection:
        raise HTTPException(
            status_code=404,
            detail=f"Collection with ID {collection_id} not found in database"
        )
    
    # Get collection name - handle both dict-like and attribute access
    collection_name = collection['name'] if isinstance(collection, dict) else collection.name
        
    # Also verify ChromaDB collection exists
    try:
        chroma_client = get_chroma_client()
        chroma_collection = chroma_client.get_collection(name=collection_name)
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_name}' exists in database but not in ChromaDB. Please recreate the collection."
        )
    
    # Prepare plugin parameters
    plugin_params = request.plugin_params or {}
    
    # Add standard parameters if not in plugin_params
    if "top_k" not in plugin_params and request.top_k is not None:
        plugin_params["top_k"] = request.top_k
    if "threshold" not in plugin_params and request.threshold is not None:
        plugin_params["threshold"] = request.threshold
    
    try:
        # Query the collection
        result = QueryService.query_collection(
            db=db,
            collection_id=collection_id,
            query_text=request.query_text,
            plugin_name=plugin_name,
            plugin_params=plugin_params
        )
        
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query collection: {str(e)}"
        )


# Root endpoint with enhanced documentation
@app.get(
    "/", 
    response_model=MessageResponse,
    summary="Root endpoint",
    description="""Returns a welcome message to confirm the server is running.
    
    Example:
    ```bash
    curl -X GET 'http://localhost:9090/' \
      -H 'Authorization: Bearer 0p3n-w3bu!'
    ```
    """,
    tags=["System"],
    responses={
        200: {"description": "Successful response with welcome message"},
        401: {"description": "Unauthorized - Invalid or missing authentication token"}
    }
)
async def root(token: str = Depends(verify_token)):
    """Root endpoint that returns a welcome message.
    
    This endpoint is primarily used to verify the server is running and authentication is working correctly.
    
    Returns:
        A dictionary containing a welcome message
    """
    return {"message": "Hello World from the Lamb Knowledge Base Server!"}


# Health check endpoint
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="""Check the health status of the server.
    
    Example:
    ```bash
    curl -X GET 'http://localhost:9090/health'
    ```
    """,
    tags=["System"],
    responses={
        200: {"description": "Server is healthy and running"}
    }
)
async def health_check():
    """Health check endpoint that does not require authentication.
    
    This endpoint can be used to verify the server is running without requiring authentication.
    
    Returns:
        A dictionary containing the server status and version
    """
    return {"status": "ok", "version": "0.1.0"}


# Database status endpoint
@app.get(
    "/database/status",
    response_model=DatabaseStatusResponse,
    summary="Database status",
    description="""Check the status of all databases.
    
    Example:
    ```bash
    curl -X GET 'http://localhost:9090/database/status' \
      -H 'Authorization: Bearer 0p3n-w3bu!'
    ```
    """,
    tags=["Database"],
    responses={
        200: {"description": "Database status information"},
        401: {"description": "Unauthorized - Invalid or missing authentication token"}
    }
)
async def database_status(token: str = Depends(verify_token), db: Session = Depends(get_db)):
    """Get the status of the SQLite and ChromaDB databases.
    
    Returns:
        A dictionary with database status information
    """
    # Re-initialize databases to get fresh status
    db_status = init_databases()
    
    # Count collections in SQLite
    collections_count = db.query(Collection).count()
    
    # Get ChromaDB collections
    chroma_client = get_chroma_client()
    chroma_collections = chroma_client.list_collections()
    
    return {
        "sqlite_status": {
            "initialized": db_status["sqlite_initialized"],
            "schema_valid": db_status["sqlite_schema_valid"],
            "errors": db_status.get("errors", [])
        },
        "chromadb_status": {
            "initialized": db_status["chromadb_initialized"],
            "collections_count": len(chroma_collections)
        },
        "collections_count": collections_count
    }


# Ingestion Plugin Endpoints

@app.get(
    "/ingestion/plugins",
    response_model=List[IngestionPluginInfo],
    summary="List ingestion plugins",
    description="""List all available document ingestion plugins.
    
    Example:
    ```bash
    curl -X GET 'http://localhost:9090/ingestion/plugins' \
      -H 'Authorization: Bearer 0p3n-w3bu!'
    ```
    """,
    tags=["Ingestion"],
    responses={
        200: {"description": "List of available ingestion plugins"},
        401: {"description": "Unauthorized - Invalid or missing authentication token"}
    }
)
async def list_ingestion_plugins(token: str = Depends(verify_token)):
    """List all available document ingestion plugins.
    
    Returns:
        List of plugin information objects
    """
    return IngestionService.list_plugins()


@app.post(
    "/collections/{collection_id}/upload",
    summary="Upload a file to a collection",
    description="""Upload a file to a collection for later ingestion.
    
    This endpoint uploads a file to the server but does not process it yet. 
    The file will be stored in the collection's directory.
    
    Example:
    ```bash
    curl -X POST 'http://localhost:9090/collections/1/upload' \
      -H 'Authorization: Bearer 0p3n-w3bu!' \
      -F 'file=@/path/to/your/document.txt'
    ```
    """,
    tags=["Ingestion"],
    responses={
        200: {"description": "File uploaded successfully"},
        401: {"description": "Unauthorized - Invalid or missing authentication token"},
        404: {"description": "Collection not found"}
    }
)
async def upload_file(
    collection_id: int,
    file: UploadFile = File(...),
    token: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Upload a file to a collection.
    
    Args:
        collection_id: ID of the collection to upload to
        file: The file to upload
        token: Authentication token
        db: Database session
        
    Returns:
        Path to the uploaded file
        
    Raises:
        HTTPException: If collection not found or upload fails
    """
    # Check if collection exists in SQLite
    collection = CollectionService.get_collection(db, collection_id)
    if not collection:
        raise HTTPException(
            status_code=404,
            detail=f"Collection with ID {collection_id} not found in database"
        )
    
    # Get collection name - handle both dict-like and attribute access
    collection_name = collection['name'] if isinstance(collection, dict) else collection.name
    collection_owner = collection['owner'] if isinstance(collection, dict) else collection.owner
        
    # Also verify ChromaDB collection exists
    try:
        chroma_client = get_chroma_client()
        chroma_collection = chroma_client.get_collection(name=collection_name)
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_name}' exists in database but not in ChromaDB. Please recreate the collection."
        )
    
    # Save the file
    try:
        file_info = IngestionService.save_uploaded_file(
            file=file,
            owner=collection_owner,
            collection_name=collection_name
        )
        
        return {
            "file_path": file_info["file_path"],
            "file_url": file_info["file_url"],
            "file_name": file.filename,
            "original_filename": file_info["original_filename"],
            "collection_id": collection_id,
            "collection_name": collection_name
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload file: {str(e)}"
        )


@app.post(
    "/collections/{collection_id}/ingest",
    response_model=IngestFileResponse,
    summary="Ingest a file with specified plugin",
    description="""Process a previously uploaded file using a specified ingestion plugin.
    
    This endpoint processes an uploaded file using the specified ingestion plugin
    but does not add the processed documents to the collection yet.
    
    Example:
    ```bash
    curl -X POST 'http://localhost:9090/collections/1/ingest' \
      -H 'Authorization: Bearer 0p3n-w3bu!' \
      -H 'Content-Type: application/json' \
      -d '{
        "file_path": "/path/to/uploaded/file.txt",
        "plugin_name": "simple_ingest",
        "plugin_params": {
          "chunk_size": 1000,
          "chunk_unit": "char",
          "chunk_overlap": 200
        }
      }'
    ```
    """,
    tags=["Ingestion"],
    responses={
        200: {"description": "File processed successfully"},
        400: {"description": "Invalid plugin parameters"},
        401: {"description": "Unauthorized - Invalid or missing authentication token"},
        404: {"description": "Collection or plugin not found"}
    }
)
async def ingest_file(
    collection_id: int,
    file_path: str = Form(...),
    plugin_name: str = Form(...),
    plugin_params: str = Form("{}"),
    token: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Ingest a file using the specified plugin.
    
    Args:
        collection_id: ID of the collection
        file_path: Path to the file to ingest
        plugin_name: Name of the plugin to use
        plugin_params: JSON string of parameters for the plugin
        token: Authentication token
        db: Database session
        
    Returns:
        List of document chunks with metadata
        
    Raises:
        HTTPException: If collection or plugin not found, or ingestion fails
    """
    # Check if collection exists in SQLite
    collection = CollectionService.get_collection(db, collection_id)
    if not collection:
        raise HTTPException(
            status_code=404,
            detail=f"Collection with ID {collection_id} not found in database"
        )
    
    # Get collection name - handle both dict-like and attribute access
    collection_name = collection['name'] if isinstance(collection, dict) else collection.name
        
    # Also verify ChromaDB collection exists
    try:
        chroma_client = get_chroma_client()
        chroma_collection = chroma_client.get_collection(name=collection_name)
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_name}' exists in database but not in ChromaDB. Please recreate the collection."
        )
    
    # Parse plugin parameters
    try:
        params = json.loads(plugin_params)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON in plugin_params"
        )
    
    # Process the file
    try:
        documents = IngestionService.ingest_file(
            file_path=file_path,
            plugin_name=plugin_name,
            plugin_params=params
        )
        
        return {
            "file_path": file_path,
            "document_count": len(documents),
            "documents": documents
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest file: {str(e)}"
        )


@app.post(
    "/collections/{collection_id}/documents",
    response_model=AddDocumentsResponse,
    summary="Add documents to a collection",
    description="""Add processed documents to a collection.
    
    This endpoint adds processed documents to a ChromaDB collection.
    
    Example:
    ```bash
    curl -X POST 'http://localhost:9090/collections/1/documents' \
      -H 'Authorization: Bearer 0p3n-w3bu!' \
      -H 'Content-Type: application/json' \
      -d '{
        "documents": [
          {
            "text": "Document content here...",
            "metadata": {
              "source": "file.txt",
              "chunk_index": 0
            }
          }
        ]
      }'
    ```
    """,
    tags=["Ingestion"],
    responses={
        200: {"description": "Documents added successfully"},
        401: {"description": "Unauthorized - Invalid or missing authentication token"},
        404: {"description": "Collection not found"}
    }
)
async def add_documents(
    collection_id: int,
    request: AddDocumentsRequest,
    token: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Add documents to a collection.
    
    Args:
        collection_id: ID of the collection
        request: Request with documents to add
        token: Authentication token
        db: Database session
        
    Returns:
        Status information about the operation
        
    Raises:
        HTTPException: If collection not found or adding documents fails
    """
    try:
        result = IngestionService.add_documents_to_collection(
            db=db,
            collection_id=collection_id,
            documents=request.documents
        )
        
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add documents to collection: {str(e)}"
        )


@app.post(
    "/collections/{collection_id}/ingest-file",
    response_model=AddDocumentsResponse,
    summary="Ingest a file directly into a collection",
    description="""Upload, process, and add a file to a collection in one operation.
    
    This endpoint combines file upload, processing with an ingestion plugin, and adding 
    to the collection in a single operation.
    
    Example for simple_ingest plugin:
    ```bash
    curl -X POST 'http://localhost:9090/collections/1/ingest-file' \
      -H 'Authorization: Bearer 0p3n-w3bu!' \
      -F 'file=@/path/to/document.txt' \
      -F 'plugin_name=simple_ingest' \
      -F 'plugin_params={"chunk_size": 1000, "chunk_unit": "char", "chunk_overlap": 200}'
    ```
    
    Parameters for simple_ingest plugin:
    - chunk_size: Size of each chunk (default: 1000)
    - chunk_unit: Unit for chunking (char, word, line) (default: char)
    - chunk_overlap: Number of units to overlap between chunks (default: 200)
    """,
    tags=["Ingestion"],
    responses={
        200: {"description": "File ingested successfully"},
        400: {"description": "Invalid plugin parameters"},
        401: {"description": "Unauthorized - Invalid or missing authentication token"},
        404: {"description": "Collection or plugin not found"},
        500: {"description": "Error processing file or adding to collection"}
    }
)
async def ingest_file_to_collection(
    collection_id: int,
    file: UploadFile = File(...),
    plugin_name: str = Form(...),
    plugin_params: str = Form("{}"),
    token: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Ingest a file directly into a collection using a specified plugin.
    
    This endpoint combines file upload, processing with the specified plugin,
    and adding to the collection in a single operation.
    
    Args:
        collection_id: ID of the collection
        file: The file to upload and ingest
        plugin_name: Name of the ingestion plugin to use
        plugin_params: JSON string of parameters for the plugin
        token: Authentication token
        db: Database session
        
    Returns:
        Status information about the ingestion operation
        
    Raises:
        HTTPException: If collection not found, plugin not found, or ingestion fails
    """
    # Check if collection exists in SQLite
    collection = CollectionService.get_collection(db, collection_id)
    if not collection:
        raise HTTPException(
            status_code=404,
            detail=f"Collection with ID {collection_id} not found in database"
        )
        
    # Get collection name - handle both dict-like and attribute access
    collection_name = collection['name'] if isinstance(collection, dict) else collection.name
        
    # Also verify ChromaDB collection exists
    try:
        chroma_client = get_chroma_client()
        chroma_collection = chroma_client.get_collection(name=collection_name)
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_name}' exists in database but not in ChromaDB. Please recreate the collection."
        )
    
    # Check if plugin exists
    plugin = IngestionService.get_plugin(plugin_name)
    if not plugin:
        raise HTTPException(
            status_code=404,
            detail=f"Ingestion plugin '{plugin_name}' not found"
        )
    
    # Parse plugin parameters
    try:
        params = json.loads(plugin_params)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON in plugin_params"
        )
    
    try:
        # Step 1: Upload file
        file_info = IngestionService.save_uploaded_file(
            file=file,
            owner=collection["owner"] if isinstance(collection, dict) else collection.owner,
            collection_name=collection_name
        )
        file_path = file_info["file_path"]
        
        # Step 2: Process file with plugin
        documents = IngestionService.ingest_file(
            file_path=file_path,
            plugin_name=plugin_name,
            plugin_params=params
        )
        
        # Step 3: Add documents to collection
        result = IngestionService.add_documents_to_collection(
            db=db,
            collection_id=collection_id,
            documents=documents
        )
        
        # Step 4: Register the file in the FileRegistry
        file_registry = IngestionService.register_file(
            db=db,
            collection_id=collection_id,
            file_path=file_path,
            file_url=file_info["file_url"],
            original_filename=file_info["original_filename"],
            plugin_name=plugin_name,
            plugin_params=params,
            owner=collection["owner"] if isinstance(collection, dict) else collection.owner,
            document_count=len(documents),
            content_type=file.content_type
        )
        
        # Add additional information to the result
        result["file_path"] = file_path
        result["file_url"] = file_info["file_url"]
        result["original_filename"] = file_info["original_filename"]
        result["plugin_name"] = plugin_name
        result["file_registry_id"] = file_registry.id
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest file: {str(e)}"
        )


# Create a new collection
@app.post(
    "/collections",
    response_model=CollectionResponse,
    summary="Create collection",
    description="""Create a new knowledge base collection.
    
    Example:
    ```bash
    curl -X POST 'http://localhost:9090/collections'       -H 'Authorization: Bearer 0p3n-w3bu!'       -H 'Content-Type: application/json'       -d '{
        "name": "my-knowledge-base",
        "description": "My first knowledge base",
        "owner": "user1",
        "visibility": "private",
        "embeddings_model": {
          "model": "default",
          "vendor": "default",
          "endpoint":"default",
          "apikey": "default"
        }
        
        # For OpenAI embeddings, use:
        # "embeddings_model": {
        #   "model": "text-embedding-3-small",
        #   "vendor": "openai",
        #   "endpoint":"https://api.openai.com/v1/embeddings"
        #   "apikey": "your-openai-key-here"
        # }
    ```
    """,
    tags=["Collections"],
    responses={
        201: {"description": "Collection created successfully"},
        400: {"description": "Bad request - Invalid collection data"},
        409: {"description": "Conflict - Collection with this name already exists"},
        401: {"description": "Unauthorized - Invalid or missing authentication token"}
    },
    status_code=status.HTTP_201_CREATED
)
async def create_collection(
    collection: CollectionCreate, 
    token: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Create a new knowledge base collection.
    
    Args:
        collection: Collection data from request body
        token: Authentication token
        db: Database session
        
    Returns:
        The created collection
        
    Raises:
        HTTPException: If collection creation fails
    """
    # Resolve default values for embeddings_model before passing to service
    if collection.embeddings_model:
        model_info = collection.embeddings_model.model_dump()
        resolved_config = {}
        
        # Resolve vendor
        vendor = model_info.get("vendor")
        if vendor == "default":
            vendor = os.getenv("EMBEDDINGS_VENDOR")
            if not vendor:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="EMBEDDINGS_VENDOR environment variable not set but 'default' specified"
                )
        resolved_config["vendor"] = vendor
        
        # Resolve model
        model = model_info.get("model")
        if model == "default":
            model = os.getenv("EMBEDDINGS_MODEL")
            if not model:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="EMBEDDINGS_MODEL environment variable not set but 'default' specified"
                )
        resolved_config["model"] = model
        
        # Resolve API key (optional)
        api_key = model_info.get("apikey")
        if api_key == "default":
            api_key = os.getenv("EMBEDDINGS_APIKEY", "")
        
        # Only log whether we have a key or not, never log the key itself or its contents
        if vendor == "openai":
            print(f"INFO: [main.create_collection] Using OpenAI API key: {'[PROVIDED]' if api_key else '[MISSING]'}")
            
        resolved_config["apikey"] = api_key
        
        # Resolve API endpoint (needed for some vendors like Ollama)
        api_endpoint = model_info.get("api_endpoint")
        if api_endpoint == "default":
            api_endpoint = os.getenv("EMBEDDINGS_ENDPOINT")
            if not api_endpoint and vendor == "ollama":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="EMBEDDINGS_ENDPOINT environment variable not set but 'default' specified for Ollama"
                )
        if api_endpoint:  # Only add if not None
            resolved_config["api_endpoint"] = api_endpoint
            
        # Log the resolved configuration
        print(f"INFO: [main.create_collection] Resolved embeddings config: {resolved_config}")
        
        # Replace default values with resolved values in the collection object
        collection.embeddings_model = EmbeddingsModel(**resolved_config)
    
    # Now call the service with default values already resolved
    return CollectionsService.create_collection(collection, db)


# List collections
@app.get(
    "/collections",
    response_model=CollectionList,
    summary="List collections",
    description="""List all available knowledge base collections.
    
    Example:
    ```bash
    curl -X GET 'http://localhost:9090/collections' \
      -H 'Authorization: Bearer 0p3n-w3bu!'
    
    # With filtering parameters
    curl -X GET 'http://localhost:9090/collections?owner=user1&visibility=public&skip=0&limit=20' \
      -H 'Authorization: Bearer 0p3n-w3bu!'
    ```
    """,
    tags=["Collections"],
    responses={
        200: {"description": "List of collections"},
        401: {"description": "Unauthorized - Invalid or missing authentication token"}
    }
)
async def list_collections(
    token: str = Depends(verify_token),
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0, description="Number of collections to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of collections to return"),
    owner: str = Query(None, description="Filter by owner"),
    visibility: str = Query(None, description="Filter by visibility ('private' or 'public')")
):
    """List all available knowledge base collections with optional filtering.
    
    Args:
        token: Authentication token
        db: Database session
        skip: Number of collections to skip
        limit: Maximum number of collections to return
        owner: Optional filter by owner
        visibility: Optional filter by visibility
        
    Returns:
        List of collections matching the filter criteria
    """
    return CollectionsService.list_collections(
        db=db,
        skip=skip,
        limit=limit,
        owner=owner,
        visibility=visibility
    )


# Get a specific collection
@app.get(
    "/collections/{collection_id}",
    response_model=CollectionResponse,
    summary="Get collection",
    description="""Get details of a specific knowledge base collection.
    
    Example:
    ```bash
    curl -X GET 'http://localhost:9090/collections/1' \
      -H 'Authorization: Bearer 0p3n-w3bu!'
    ```
    """,
    tags=["Collections"],
    responses={
        200: {"description": "Collection details"},
        404: {"description": "Not found - Collection not found"},
        401: {"description": "Unauthorized - Invalid or missing authentication token"}
    }
)
async def get_collection(
    collection_id: int,
    token: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Get details of a specific knowledge base collection.
    
    Args:
        collection_id: ID of the collection to retrieve
        token: Authentication token
        db: Database session
        
    Returns:
        Collection details
        
    Raises:
        HTTPException: If collection not found
    """
    return CollectionsService.get_collection(collection_id, db)


# File Registry Endpoints
@app.get(
    "/collections/{collection_id}/files",
    response_model=List[FileRegistryResponse],
    summary="List files in a collection",
    description="Get a list of all files in a collection",
    tags=["Files"],
    responses={
        200: {"description": "List of files in the collection"},
        401: {"description": "Unauthorized - Invalid or missing authentication token"},
        404: {"description": "Collection not found"},
        500: {"description": "Server error"}
    }
)
async def list_files(
    collection_id: int,
    token: str = Depends(verify_token),
    db: Session = Depends(get_db),
    status: str = Query(None, description="Filter by status (completed, processing, failed, deleted)")
):
    """List all files in a collection.
    
    Args:
        collection_id: ID of the collection
        token: Authentication token
        db: Database session
        status: Optional filter by status
        
    Returns:
        List of file registry entries
        
    Raises:
        HTTPException: If collection not found
    """
    return CollectionsService.list_files(collection_id, db, status)


@app.put(
    "/files/{file_id}/status",
    response_model=FileRegistryResponse,
    summary="Update file status",
    description="Update the status of a file in the registry",
    tags=["Files"],
    responses={
        200: {"description": "File status updated successfully"},
        401: {"description": "Unauthorized - Invalid or missing authentication token"},
        404: {"description": "File not found"},
        500: {"description": "Server error"}
    }
)
async def update_file_status(
    file_id: int,
    status: str = Query(..., description="New status (completed, processing, failed, deleted)"),
    token: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Update the status of a file in the registry.
    
    Args:
        file_id: ID of the file registry entry
        status: New status
        token: Authentication token
        db: Database session
        
    Returns:
        Updated file registry entry
        
    Raises:
        HTTPException: If file not found or status invalid
    """
    return CollectionsService.update_file_status(file_id, status, db)

