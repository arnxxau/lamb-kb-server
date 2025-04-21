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

from fastapi import Depends, FastAPI, HTTPException, status, Query, File, Form, UploadFile, BackgroundTasks
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
from services.collections import CollectionsService
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
    IngestURLRequest,
    IngestURLResponse,
    AddDocumentsRequest,
    AddDocumentsResponse,
    PreviewURLRequest, 
    PreviewURLResponse
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
    token: str = Depends(verify_token)
):
    """Query a collection using a specified plugin.
    
    Args:
        collection_id: ID of the collection to query
        request: Query request parameters
        plugin_name: Name of the query plugin to use
        token: Authentication token
        
    Returns:
        Query results
        
    Raises:
        HTTPException: If collection not found, plugin not found, or query fails
    """
    # Get collection from service layer
    try:
        collection = CollectionsService.get_collection(collection_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Collection with ID {collection_id} not found in database: {str(e)}"
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
    # Get collection from service layer
    try:
        collection = CollectionsService.get_collection(collection_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Collection with ID {collection_id} not found in database: {str(e)}"
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
    # Get collection from service layer
    try:
        collection = CollectionsService.get_collection(collection_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Collection with ID {collection_id} not found in database: {str(e)}"
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
    "/collections/{collection_id}/ingest-url",
    response_model=AddDocumentsResponse,
    summary="Ingest content from URLs directly into a collection",
    description="""Fetch, process, and add content from URLs to a collection in one operation.

    This endpoint fetches content from specified URLs, processes it with the URL ingestion plugin,
    and adds the content to the collection.

    Example:
    ```bash
    curl -X POST 'http://localhost:9090/collections/1/ingest-url' \
      -H 'Authorization: Bearer 0p3n-w3bu!' \
      -H 'Content-Type: application/json' \
      -d '{
        "urls": ["https://example.com/page1", "https://example.com/page2"],
        "plugin_params": {
          "chunk_size": 1000,
          "chunk_unit": "char",
          "chunk_overlap": 200
        }
      }'
    ```

    Parameters for url_ingest plugin:
    - urls: List of URLs to ingest
    - chunk_size: Size of each chunk (default: 1000)
    - chunk_unit: Unit for chunking (char, word, line) (default: char)
    - chunk_overlap: Number of units to overlap between chunks (default: 200)
    """,
    tags=["Ingestion"],
    responses={
        200: {"description": "URLs ingested successfully"},
        400: {"description": "Invalid plugin parameters or URLs"},
        401: {"description": "Unauthorized - Invalid or missing authentication token"},
        404: {"description": "Collection or plugin not found"},
        500: {"description": "Error processing URLs or adding to collection"}
    }
)
async def ingest_url_to_collection(
    collection_id: int,
    request: IngestURLRequest,
    token: str = Depends(verify_token),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None
):
    """Ingest content from URLs directly into a collection.

    This endpoint fetches content from specified URLs, processes it with the URL ingestion plugin,
    and adds the content to the collection.

    Args:
        collection_id: ID of the collection
        request: Request with URLs and processing parameters
        token: Authentication token
        db: Database session
        background_tasks: FastAPI background tasks
        
    Returns:
        Status information about the ingestion operation
        
    Raises:
        HTTPException: If collection not found, plugin not found, or ingestion fails
    """
    # Get collection from service layer
    try:
        collection = CollectionsService.get_collection(collection_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Collection with ID {collection_id} not found in database: {str(e)}"
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
    plugin_name = request.plugin_name
    plugin = IngestionService.get_plugin(plugin_name)
    if not plugin:
        raise HTTPException(
            status_code=404,
            detail=f"Ingestion plugin '{plugin_name}' not found"
        )
    
    # Check if URLs are provided
    if not request.urls:
        raise HTTPException(
            status_code=400,
            detail="No URLs provided"
        )
    
    try:
        # Create a temporary file to track this URL ingestion
        import tempfile
        import uuid
        import os
        
        temp_dir = os.path.join(tempfile.gettempdir(), "url_ingestion")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}.url")
        
        # Write URLs to the temporary file (just for tracking)
        with open(temp_file_path, "w") as f:
            for url in request.urls:
                f.write(f"{url}\n")
        
        # Step 1: Register the URL ingestion in the FileRegistry with PROCESSING status
        # Store the first URL as the filename to make it easier to display and preview
        first_url = request.urls[0] if request.urls else "unknown_url"
        file_registry = IngestionService.register_file(
            db=db,
            collection_id=collection_id,
            file_path=temp_file_path,
            file_url=first_url,  # Store the URL for direct access
            original_filename=first_url,  # Use the first URL as the original filename for better display
            plugin_name="url_ingest",  # Ensure consistent plugin name
            plugin_params={"urls": request.urls, **request.plugin_params},
            owner=collection["owner"] if isinstance(collection, dict) else collection.owner,
            document_count=0,  # Will be updated after processing
            content_type="text/plain",
            status=FileStatus.PROCESSING  # Set initial status to PROCESSING
        )
        
        # Step 2: Schedule background task for processing and adding documents
        def process_urls_in_background(urls: List[str], plugin_name: str, params: dict, 
                                   collection_id: int, file_registry_id: int):
            try:
                print(f"DEBUG: [background_task] Started processing URLs: {', '.join(urls[:3])}...")
                
                # Create a new session for the background task
                from database.connection import SessionLocal
                db_background = SessionLocal()
                
                try:
                    # Make a placeholder file path for the URL ingestion
                    import tempfile
                    temp_file = tempfile.NamedTemporaryFile(delete=False)
                    temp_file_path = temp_file.name
                    temp_file.close()
                    
                    # Step 2.1: Process URLs with plugin
                    # Add URLs to the plugin parameters
                    full_params = {**params, "urls": urls}
                    
                    documents = IngestionService.ingest_file(
                        file_path=temp_file_path,  # This is just a placeholder
                        plugin_name=plugin_name,
                        plugin_params=full_params
                    )
                    
                    # Step 2.2: Add documents to collection
                    result = IngestionService.add_documents_to_collection(
                        db=db_background,
                        collection_id=collection_id,
                        documents=documents
                    )
                    
                    # Step 2.3: Update file registry with completed status and document count
                    IngestionService.update_file_status(
                        db=db_background, 
                        file_id=file_registry_id, 
                        status=FileStatus.COMPLETED
                    )
                    
                    # Update document count
                    file_reg = db_background.query(FileRegistry).filter(FileRegistry.id == file_registry_id).first()
                    if file_reg:
                        file_reg.document_count = len(documents)
                        db_background.commit()
                    
                    # Clean up the temporary file
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass
                    
                    print(f"DEBUG: [background_task] Completed processing URLs with {len(documents)} documents")
                finally:
                    db_background.close()
                    
            except Exception as e:
                print(f"ERROR: [background_task] Failed to process URLs: {str(e)}")
                import traceback
                print(f"ERROR: [background_task] Stack trace:\n{traceback.format_exc()}")
                
                # Update file status to FAILED
                try:
                    from database.connection import SessionLocal
                    db_error = SessionLocal()
                    try:
                        IngestionService.update_file_status(
                            db=db_error, 
                            file_id=file_registry_id, 
                            status=FileStatus.FAILED
                        )
                    finally:
                        db_error.close()
                except Exception:
                    print(f"ERROR: [background_task] Could not update file status to FAILED")
        
        # Add the task to background tasks
        background_tasks.add_task(
            process_urls_in_background, 
            request.urls, 
            plugin_name, 
            request.plugin_params, 
            collection_id, 
            file_registry.id
        )
        
        # Return immediate response with URL information
        return {
            "collection_id": collection_id,
            "collection_name": collection_name,
            "documents_added": 0,  # Initially 0 since processing will happen in background
            "success": True,
            "file_path": temp_file_path,
            "file_url": "",
            "original_filename": f"urls_{len(request.urls)}",
            "plugin_name": plugin_name,
            "file_registry_id": file_registry.id,
            "status": "processing"
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest URLs: {str(e)}"
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
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None
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
        background_tasks: FastAPI background tasks
        
    Returns:
        Status information about the ingestion operation
        
    Raises:
        HTTPException: If collection not found, plugin not found, or ingestion fails
    """
    # Get collection from service layer
    try:
        collection = CollectionsService.get_collection(collection_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Collection with ID {collection_id} not found in database: {str(e)}"
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
        # Step 1: Upload file (this step remains synchronous)
        file_info = IngestionService.save_uploaded_file(
            file=file,
            owner=collection["owner"] if isinstance(collection, dict) else collection.owner,
            collection_name=collection_name
        )
        file_path = file_info["file_path"]
        file_url = file_info["file_url"]
        original_filename = file_info["original_filename"]
        owner = collection["owner"] if isinstance(collection, dict) else collection.owner
        
        # Step 2: Register the file in the FileRegistry with PROCESSING status
        file_registry = IngestionService.register_file(
            db=db,
            collection_id=collection_id,
            file_path=file_path,
            file_url=file_url,
            original_filename=original_filename,
            plugin_name=plugin_name,
            plugin_params=params,
            owner=owner,
            document_count=0,  # Will be updated after processing
            content_type=file.content_type,
            status=FileStatus.PROCESSING  # Set initial status to PROCESSING
        )
        
        # Step 3: Schedule background task for processing and adding documents
        def process_file_in_background(file_path: str, plugin_name: str, params: dict, 
                                     collection_id: int, file_registry_id: int):
            try:
                print(f"DEBUG: [background_task] Started processing file: {file_path}")
                
                # Create a new session for the background task
                from database.connection import SessionLocal
                db_background = SessionLocal()
                
                try:
                    # Step 3.1: Process file with plugin
                    documents = IngestionService.ingest_file(
                        file_path=file_path,
                        plugin_name=plugin_name,
                        plugin_params=params
                    )
                    
                    # Step 3.2: Add documents to collection
                    result = IngestionService.add_documents_to_collection(
                        db=db_background,
                        collection_id=collection_id,
                        documents=documents
                    )
                    
                    # Step 3.3: Update file registry with completed status and document count
                    IngestionService.update_file_status(
                        db=db_background, 
                        file_id=file_registry_id, 
                        status=FileStatus.COMPLETED
                    )
                    
                    # Update document count
                    file_reg = db_background.query(FileRegistry).filter(FileRegistry.id == file_registry_id).first()
                    if file_reg:
                        file_reg.document_count = len(documents)
                        db_background.commit()
                    
                    print(f"DEBUG: [background_task] Completed processing file {file_path} with {len(documents)} documents")
                finally:
                    db_background.close()
                    
            except Exception as e:
                print(f"ERROR: [background_task] Failed to process file {file_path}: {str(e)}")
                import traceback
                print(f"ERROR: [background_task] Stack trace:\n{traceback.format_exc()}")
                
                # Update file status to FAILED
                try:
                    from database.connection import SessionLocal
                    db_error = SessionLocal()
                    try:
                        IngestionService.update_file_status(
                            db=db_error, 
                            file_id=file_registry_id, 
                            status=FileStatus.FAILED
                        )
                    finally:
                        db_error.close()
                except Exception:
                    print(f"ERROR: [background_task] Could not update file status to FAILED")
        
        # Add the task to background tasks
        background_tasks.add_task(
            process_file_in_background, 
            file_path, 
            plugin_name, 
            params, 
            collection_id, 
            file_registry.id
        )
        
        # Return immediate response with file information
        return {
            "collection_id": collection_id,
            "collection_name": collection_name,
            "documents_added": 0,  # Initially 0 since processing will happen in background
            "success": True,
            "file_path": file_path,
            "file_url": file_url,
            "original_filename": original_filename,
            "plugin_name": plugin_name,
            "file_registry_id": file_registry.id,
            "status": "processing"
        }
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
    token: str = Depends(verify_token)
):
    """Create a new knowledge base collection.
    
    Args:
        collection: Collection data from request body
        token: Authentication token
        
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
    return CollectionsService.create_collection(collection)


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
    skip: int = Query(0, ge=0, description="Number of collections to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of collections to return"),
    owner: str = Query(None, description="Filter by owner"),
    visibility: str = Query(None, description="Filter by visibility ('private' or 'public')")
):
    """List all available knowledge base collections with optional filtering.
    
    Args:
        token: Authentication token
        skip: Number of collections to skip
        limit: Maximum number of collections to return
        owner: Optional filter by owner
        visibility: Optional filter by visibility
        
    Returns:
        List of collections matching the filter criteria
    """
    return CollectionsService.list_collections(
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
    token: str = Depends(verify_token)
):
    """Get details of a specific knowledge base collection.
    
    Args:
        collection_id: ID of the collection to retrieve
        token: Authentication token
        
    Returns:
        Collection details
        
    Raises:
        HTTPException: If collection not found
    """
    return CollectionsService.get_collection(collection_id)


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
    status: str = Query(None, description="Filter by status (completed, processing, failed, deleted)")
):
    """List all files in a collection.
    
    Args:
        collection_id: ID of the collection
        token: Authentication token
        status: Optional filter by status
        
    Returns:
        List of file registry entries
        
    Raises:
        HTTPException: If collection not found
    """
    return CollectionsService.list_files(collection_id, status)



@app.get(
    "/files/{file_id}/content",
    summary="Get file content",
    description="Get the content of a file from the collection",
    tags=["Files"],
    responses={
        200: {"description": "File content retrieved successfully"},
        401: {"description": "Unauthorized - Invalid or missing authentication token"},
        404: {"description": "File not found"},
        500: {"description": "Server error"}
    }
)
async def get_file_content(
    file_id: int,
    token: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Get the content of a file.
    
    This is useful for previewing the content of a file, especially for URLs that have been ingested.
    
    Args:
        file_id: ID of the file registry entry
        token: Authentication token
        db: Database session
        
    Returns:
        Content of the file
        
    Raises:
        HTTPException: If file not found or content cannot be retrieved
    """
    try:
        # Get file registry entry
        file_registry = db.query(FileRegistry).filter(FileRegistry.id == file_id).first()
        if not file_registry:
            raise HTTPException(
                status_code=404,
                detail=f"File with ID {file_id} not found"
            )
        
        # Get the file content based on the plugin type
        file_content = ""
        content_type = "text"
        
        # Get collection for any file type
        collection = db.query(Collection).filter(Collection.id == file_registry.collection_id).first()
        if not collection:
            raise HTTPException(
                status_code=404,
                detail=f"Collection with ID {file_registry.collection_id} not found"
            )
            
        # Get database content from ChromaDB
        from database.connection import get_chroma_client
        
        # Get ChromaDB client and collection
        chroma_client = get_chroma_client()
        chroma_collection = chroma_client.get_collection(name=collection.name)
        
        # Get the source filename/url
        source = file_registry.original_filename
        
        # Get the file type and plugin name
        plugin_name = file_registry.plugin_name
        
        # Set the appropriate file type based on extension or plugin
        original_extension = ""
        if "." in source:
            original_extension = source.split(".")[-1].lower()
            
        # Determine content type based on file extension or plugin name
        if plugin_name == "url_ingest" or source.startswith(("http://", "https://")):
            content_type = "markdown"  # URL content from firecrawl is markdown
        elif original_extension in ["md", "markdown"]:
            content_type = "markdown"
        elif original_extension in ["txt", "text"]:
            content_type = "text"
        else:
            content_type = "text"  # Default to text
            
        # Query ChromaDB for documents with this source
        results = chroma_collection.get(
            where={"source": source}, 
            include=["documents", "metadatas"]
        )
        
        if not results or not results["documents"] or len(results["documents"]) == 0:
            # Try other fields if "source" doesn't work
            results = chroma_collection.get(
                where={"filename": source}, 
                include=["documents", "metadatas"]
            )
            
        if not results or not results["documents"] or len(results["documents"]) == 0:
            # For regular files, the source field might be the file path
            # Try to find by just the filename part
            import os
            filename = os.path.basename(source)
            if filename:
                results = chroma_collection.get(
                    where={"filename": filename}, 
                    include=["documents", "metadatas"]
                )
                
        if not results or not results["documents"] or len(results["documents"]) == 0:
            # If we still haven't found it, check if there's a physical file we can read
            if os.path.exists(file_registry.file_path) and os.path.isfile(file_registry.file_path):
                try:
                    with open(file_registry.file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                        
                    return {
                        "file_id": file_id,
                        "original_filename": source,
                        "content": file_content,
                        "content_type": content_type,
                        "chunk_count": 1,
                        "timestamp": file_registry.updated_at.isoformat() if file_registry.updated_at else None
                    }
                except Exception as e:
                    print(f"Error reading file from disk: {str(e)}")
                    # Continue to error handling
            
            raise HTTPException(
                status_code=404,
                detail=f"No content found for file: {source}"
            )
        
        # Reconstruct the content from chunks
        # First sort by chunk_index
        chunk_docs = []
        for i, doc in enumerate(results["documents"]):
            if i < len(results["metadatas"]) and results["metadatas"][i]:
                metadata = results["metadatas"][i]
                chunk_docs.append({
                    "text": doc,
                    "index": metadata.get("chunk_index", i),
                    "count": metadata.get("chunk_count", 0)
                })
        
        # Sort chunks by index
        chunk_docs.sort(key=lambda x: x["index"])
        
        # Join all chunks
        full_content = "\n".join(doc["text"] for doc in chunk_docs)
        
        # Return content with metadata
        return {
            "file_id": file_id,
            "original_filename": source,
            "content": full_content,
            "content_type": content_type,
            "chunk_count": len(chunk_docs),
            "timestamp": file_registry.updated_at.isoformat() if file_registry.updated_at else None
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        import traceback
        print(f"Error retrieving file content: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve file content: {str(e)}"
        )


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
    token: str = Depends(verify_token)
):
    """Update the status of a file in the registry.
    
    Args:
        file_id: ID of the file registry entry
        status: New status
        token: Authentication token
        
    Returns:
        Updated file registry entry
        
    Raises:
        HTTPException: If file not found or status invalid
    """
    return CollectionsService.update_file_status(file_id, status)